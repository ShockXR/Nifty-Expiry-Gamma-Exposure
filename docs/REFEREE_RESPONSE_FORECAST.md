# Response to the referee: forecasting (M4, M5, minor comment 15)

**REAL DATA. POST HOC.** Every analysis in this note was done after the referee report, in response to it. It uses the committed forecasts of the final run (`results/pipeline_nse_v2/`, 32 models, 792 test days) and, for the new benchmarks, the same NSE data and the same 21-day walk-forward. The pipeline code and its committed results are unchanged. Every number below is copied from a file in `results/referee_forecast/`, from `results/referee_forecast_stdout.txt` (the printed log of the run), or from a committed results file named next to it.

Code: `experiments/referee_forecast.py`. Tests on synthetic data only: `tests/test_referee_forecast.py`.

## Summary

| Referee point | What we did | Short answer | Claim changes? |
|---|---|---|---|
| M4.1 best of 32 | Hansen SPA, White Reality Check, step-down adjusted p-values | Some challenger beats mcsGARCH in every horizon and subset (SPA p at most 0.040). The hybrid survives the adjustment in 8 of 9 cells; not in the last hour on expiry days (adjusted p 0.062) | Yes: drop "only model with p at most 0.010 in all six columns" and "also on expiry days" for the last hour |
| M4.3 estimation noise | Seed noise added to the DM variance; spread of 6 hybrid variants; cost of a full rerun | Seed noise is 0.19 points against margins of 5.5 to 9.2 points; p-values barely move. A full seed bootstrap needs a rerun (at most 6.3 hours on one core); not done | Small: quote noise-adjusted p-values, not 1.8e-9 |
| M5 fair benchmark | mcsGARCH given VIX or the straddle (4 variants, 3 of them new), ridge on the same features | At 15 and 30 minutes the hybrid still wins by 4.8% (p 0.005) and 5.9% (p below 0.001). In the last hour it does not: mcsGARCH with VIX as its daily level is within 1.7% (p 0.227) | Yes: the last-hour gain is information, not method |
| M4.2 and minor 15 MCS | Recomputed the MCS (10,000 replications, two block lengths) and listed what it keeps | Last hour: keeps VIX and HAR on all days, mcsGARCH on expiry days. Top-ranked: `hybrid_gex_us`, `hybrid_v1gex`, `linear_v1`; never `hybrid_base` | Yes: say this plainly |
| M4.4 hold-out | Hold-out table and DM tests of LightGBM against the hybrid | LightGBM is ahead in the hold-out, behind in development, and never significantly different over all days | Yes: drop "needs the hybrid structure" |
| M5 MSE | Ranking under MSE from committed files | MSE ranking is driven by about ten windows; no MSE ranking is reliable | Wording only |
| Side finding | The benchmark's EWMA fallback looked at later days | Excluding the 42 affected days, the hybrid's margin is 5.9%, 7.9%, 6.5% | Disclose |
| Priority 5 (added later) | Giacomini and White unconditional and conditional tests, 42 p-values plus 12 robustness | The advantage does not concentrate on expiry days (p 0.234 to 0.883). Over the matched benchmark at 15 and 30 minutes it comes from the periods since November 2024 (multi-weekly period: 0.2%, 0.7%) | One added sentence |

## Terms used (for a beginner)

* **QLIKE.** The loss used to score a variance forecast f against the realized variance y: y/f - log(y/f) - 1. Zero is perfect. It punishes forecasts that are too low more than forecasts that are too high, and every window counts in proportion (Patton 2011).
* **Margin.** How much lower a model's average daily QLIKE is than mcsGARCH's, in % of mcsGARCH's. "-5.8%" means 5.8% lower loss. "Points" means percentage points of mcsGARCH's QLIKE.
* **mcsGARCH.** The model of Chhimwal, Pandey and Pandey (CPP), a multiplicative component GARCH (Engle and Sokalska 2012): a daily variance from an EGARCH on daily returns, times a time-of-day curve, times an intraday GARCH. CPP fitted it but made no forecasts; the forecasts here are our implementation.
* **Hybrid (`hybrid_base`).** mcsGARCH times exp(a LightGBM correction). The correction is learned from the previous evening's option chain, India VIX, calendar features and recent realized variance.
* **DM test** (Diebold and Mariano 1995). Is the average loss difference between two models bigger than chance would produce? We use daily average losses, a HAC standard error and the Harvey, Leybourne and Newbold (1997) small-sample correction. The p-value is the probability of a difference this large if the two models were equally good.
* **Data snooping.** If you try 31 models and report the best, its p-value is too small: one of 31 will look good by luck. The SPA test and the Reality Check ask "does the best of the 31 really beat the benchmark, after allowing for the search?"
* **Stationary bootstrap** (Politis and Romano 1994). Resampling days in blocks of random length (mean 10 days here), so that the dependence between neighbouring days is kept.
* **Step-down adjusted p-value** (Romano and Wolf 2005). A p-value for one model that already allows for the 31 comparisons.
* **Model Confidence Set, MCS** (Hansen, Lunde and Nason 2011). The set of models that cannot be told apart from the best one at a 10% level. A wide set means the data cannot rank the models.
* **Information-matched benchmark.** A version of mcsGARCH that is given the same option-implied or VIX information as the hybrid, so that a win measures the method, not the extra information.
* **Seed.** The random number that LightGBM uses. Different seeds give slightly different models.

## M4.1. "Best of 32 models": Hansen's SPA and White's Reality Check

**What the referee said.** "Of 32 models it is the only one with p at most 0.010 against mcsGARCH in all six columns" is a selection statement; report the Reality Check or the SPA test.

**What we did.** For each horizon (next 15 minutes, next 30 minutes, last hour) and each subset (all days, expiry days, days before the closing auction) we took the daily average QLIKE of mcsGARCH and of the 31 challengers. The null hypothesis is "no challenger has lower expected loss than mcsGARCH". We computed:

* Hansen's (2005) SPA test: the largest studentized loss difference, with his lower, consistent and upper p-values (the consistent one is the recommended test);
* White's (2000) Reality Check: the same with the raw (not studentized) loss difference;
* for every challenger, single-step and step-down max-t adjusted p-values (Romano and Wolf 2005).

All use the stationary bootstrap with mean block 10 days and 10,000 replications, with blocks of 5 and 20 days as a check. The code is our own numpy implementation, because the SPA in the arch package (version 8.0.0) does not studentize. The tests check it against arch where they should agree (the bootstrap variance, and the Reality Check against arch's "upper" p-value) and on simulated data with a known answer (size under the null, power, and the case below).

**Result (REAL DATA; `results/referee_forecast/spa_reality_check.csv`, `spa_adjusted_pvalues_by_model.csv`).**

| Horizon | Subset | Days | Best challenger (by t) | Its margin | SPA consistent p | Reality Check p | `hybrid_base` margin | step-down p |
|---|---|---|---|---|---|---|---|---|
| next 15 | all | 792 | hybrid_v1gex | -5.6% | below 0.001 | 0.662 | -5.8% | below 0.001 |
| next 15 | expiry | 169 | hybrid_v1 | -5.8% | 0.006 | 0.762 | -6.6% | 0.007 |
| next 15 | pre-auction | 729 | hybrid_gex_us | -5.8% | below 0.001 | 0.677 | -5.5% | below 0.001 |
| next 30 | all | 792 | hybrid_v1gex | -8.1% | below 0.001 | 0.534 | -7.6% | below 0.001 |
| next 30 | expiry | 169 | hybrid_gex_participant | -7.0% | below 0.001 | 0.672 | -6.2% | 0.001 |
| next 30 | pre-auction | 729 | hybrid_gex_client | -6.9% | below 0.001 | 0.561 | -6.9% | below 0.001 |
| last hour | all | 792 | hybrid_gex_placebo | -5.8% | 0.003 | 0.757 | -6.4% | 0.005 |
| last hour | expiry | 169 | hybrid_gex_gross | -7.5% | 0.040 | 0.821 | -7.9% | 0.062 |
| last hour | pre-auction | 729 | hybrid_gex_placebo | -7.1% | 0.006 | 0.655 | -7.0% | 0.010 |

With blocks of 5 or 20 days the SPA p-values hardly move, except in the last hour on expiry days (0.057 with 5-day blocks, 0.039 with 20-day blocks). Counting challengers that beat mcsGARCH at 5% after the step-down adjustment: 15, 16 and 15 at 15 minutes; 18, 17 and 14 at 30 minutes; 10, 0 and 8 in the last hour (all days, expiry days, pre-auction).

**Why the SPA rejects and the Reality Check does not.** The Reality Check uses raw loss differences. The ladder contains naive models whose loss differences are very noisy (`naive_prevday` has QLIKE 238% above mcsGARCH at 15 minutes, `scoreboard_next15.csv`). Without studentizing, the bootstrap maximum tends to come from these noisy models, so the Reality Check's critical value is set by the worst models, and it loses power. This is exactly the problem Hansen (2005) designed the SPA to fix by studentizing each difference. The synthetic test `test_spa_is_not_fooled_by_poor_noisy_models_but_rc_is` reproduces it: one model that is truly better plus 30 poor, noisy ones; the SPA rejects and the Reality Check does not. So we read the SPA, and we report the Reality Check because the referee asked for it.

**What it changes.**

* The forecasting result is not an artefact of searching over 31 models: at 15 and 30 minutes, and in the last hour on all days and before the auction, some challenger beats mcsGARCH even after the search is accounted for, and `hybrid_base` itself does (step-down p at most 0.010).
* In the last hour on expiry days no challenger survives the step-down adjustment (0 of 31), and `hybrid_base` has adjusted p 0.062. So "also on expiry days" must be dropped for the last hour, and the sentence "the only one with p at most 0.010 in all six columns" should be replaced by the SPA and step-down results.
* Limitation: the SPA covers the 31 challengers of the final run. Models tried and dropped in earlier iterations (`docs/MODEL_ITERATIONS_REAL.md`) are not in it, and the choice of the hybrid as the headline was made after the first full-sample run. No test on these data can undo that.

## M4.3. Estimation noise in the DM p-values

**What the referee said.** The DM p-values treat fitted models as fixed, yet two seeds of the same hybrid differ with DM p 0.024. Use the Giacomini and White (2006) test or a bootstrap over seeds and refits.

**What we did.** The committed files contain three single-seed runs of four models at 30 minutes (`results/pipeline_nse_v2/seed_noise_next30.csv`); the headline forecasts average 5 seeds. That is not enough for a bootstrap over seeds, so we did three things:

1. Treated seed noise as an extra, independent error in the measured margin and added its variance to the DM variance: t = d / sqrt(se² + σ²). We tried several values of σ, from the measured one up to a pessimistic 95% upper bound scaled up for subsets with fewer forecasts, and computed the **break-even σ**: how large the seed noise would have to be for p to reach 5% or 1%.
2. Used the six hybrid variants that differ only by feature blocks the ablation found uninformative (GEX sets and a placebo) as a second measure of how much refitting moves the result.
3. Estimated what a real bootstrap over seeds would cost to run.

On Giacomini and White (2006): their unconditional test uses the same statistic as the DM test on the realized losses, but its theory treats the whole forecasting method (including estimation) as the object and needs a rolling estimation window of fixed length. Our models are refitted on all past days (an expanding window), so the formal conditions do not hold. We therefore report noise-adjusted DM p-values here. The unconditional and conditional Giacomini and White tests, run with that caveat, are in the section "Referee priority 5" below.

**Result (REAL DATA; `results/referee_forecast/seed_noise_summary.csv`, `seed_adjusted_dm.csv`, `refit_envelope.csv`, `seed_rerun_cost.csv`).**

* Seed noise. At 30 minutes the three single-seed hybrids have margins of -7.46%, -7.83% and -7.63%: standard deviation 0.19 points. Pooled over the four models it is 0.16 points (8 degrees of freedom), with a 95% upper bound of 0.28 points.
* Why the two seeds "differ with p 0.024". Their QLIKE differs by 0.40% (`seed_noise_next30.csv`, seeds 7 and 8). Two runs of the same model make almost the same forecast on every window, so even a tiny difference has a tiny standard error and can be "significant". The p-value says the seed matters a little; the size of the difference says how little.
* Break-even noise. Against the hybrid's margins over mcsGARCH:

| Horizon | Subset | Margin | DM SE (points) | DM p | p with pessimistic noise | Break-even σ for p = 5% (points) | for p = 1% |
|---|---|---|---|---|---|---|---|
| next 15 | all | -5.8% | 0.97 | below 0.001 | below 0.001 | 3.38 | 2.29 |
| next 15 | expiry | -6.6% | 2.12 | 0.001 | 0.001 | 3.41 | 1.87 |
| next 15 | hold-out | -6.4% | 2.04 | below 0.001 | 0.001 | 3.29 | 1.81 |
| next 30 | all | -7.6% | 0.96 | below 0.001 | below 0.001 | 4.54 | 3.14 |
| next 30 | expiry | -6.2% | 1.73 | below 0.001 | below 0.001 | 3.35 | 2.01 |
| next 30 | hold-out | -9.2% | 1.81 | below 0.001 | below 0.001 | 5.30 | 3.51 |
| last hour | all | -6.4% | 1.71 | below 0.001 | 0.002 | 3.47 | 2.13 |
| last hour | expiry | -7.9% | 3.35 | 0.010 | 0.039 | 3.41 | 0.34 |
| last hour | hold-out | -7.1% | 1.83 | below 0.001 | 0.006 | 3.90 | 2.42 |

("Pessimistic noise" is the 95% upper bound of 0.28 points, scaled up by the square root of the ratio of forecast counts for smaller subsets; the pre-auction rows are in the file.) To push any of these p-values above 5%, the seed noise would have to be at least 3.29 points, more than ten times its 95% upper bound. To push one above 1% it would take at least 1.81 points, except in the last hour on expiry days, where 0.34 points would do.

* Refit spread. Across the six hybrid variants the margin moves by at most 0.88 points (SD; last hour, hold-out) and the largest DM p in any cell is 0.037 (last hour, expiry days).
* A real bootstrap over seeds and refits would rerun the walk-forward with new seeds. From the final run's own timing (`run_info.json`, `hyperparameters_*.csv`) and the measured mcsGARCH fit time, 20 seed configurations for the hybrid, plain and tuned LightGBM at all three horizons take at most 6.3 hours on one core (the hybrid alone at most 2.2 hours). **We did not run it.**

**What it changes.** The conclusions do not change. The paper should quote the noise-adjusted p-values (or say "below 0.001"), not 1.8e-9, and should say that a full bootstrap over seeds was not run. The last hour on expiry days is again the fragile cell.

## M5. A fair benchmark: mcsGARCH given the same information (mcsGARCH-X)

**What the referee said.** The hybrid gets the previous evening's option chain, VIX and calendar features; mcsGARCH gets none. Augment mcsGARCH with the same information, or rephrase the claim. The benchmark also fell back to an EWMA in 2 of 38 refits.

**What we did.** Four information-matched versions of mcsGARCH, all refitted on the same 38 walk-forward blocks:

* `mcsgarch_iv` (committed): the daily level is the straddle-implied variance from the previous evening's chain.
* `mcsgarch_vix` (new): the daily level is India VIX at the previous close, as a daily variance (VIX²/252).
* `mcsgarchx_vix` (new): the paper's daily AR(1)-EGARCH(1,1) with Student t errors (Nelson 1991) plus log VIX at the previous close in the log-variance equation (an EGARCH-X).
* `mcsgarchx_vix_iv` (new): the same with log VIX and the log straddle-implied variance.

The EGARCH-X is our own numpy code, tested to reproduce the arch package's EGARCH exactly when there is no extra regressor, to find the true coefficient on simulated data and to use no future data. It is fitted on training days only, from several starting points, and uses a strictly real-time rule: if a test-day variance is 5 log points or more away from the training variance, it switches to the paper's daily variance from that day on. We also built the two simple competitors the referee named: `linear_base` (a ridge regression on the hybrid's features, already in the ladder) and `best_matched_mcsgarch`, which in each subset takes whichever of the four matched variants has the lowest average loss. That choice is made after seeing the results, so it favours the benchmark.

The whole build took 144 seconds (`mcsgarchx_run_info.json`). Our rebuild of mcsGARCH matches the committed forecasts to a relative difference of 7.6e-13 or less.

**Result (REAL DATA; `results/referee_forecast/fair_benchmarks.csv`, `mcsgarchx_forecasts_*.csv`, `mcsgarchx_params.csv`).**

VIX matters for the daily variance in sample: its coefficient has median 0.36 across refits, and the likelihood ratio against the plain EGARCH exceeds 3.84 (the 5% critical value) in 34 of 38 refits; with both regressors it exceeds 5.99 in 33 of 38. Out of sample, the matched benchmarks against mcsGARCH (QLIKE margin, one-sided p that the benchmark is better):

| Horizon, all days | mcsgarch_iv | mcsgarch_vix | mcsgarchx_vix | mcsgarchx_vix_iv | linear_base |
|---|---|---|---|---|---|
| next 15 | -1.0% (0.250) | -0.6% (0.159) | +0.5% (0.725) | -0.5% (0.347) | -2.2% (0.104) |
| next 30 | -1.9% (0.142) | -1.2% (0.036) | +0.7% (0.738) | -0.7% (0.355) | -3.9% (0.013) |
| last hour | -1.2% (0.345) | -4.8% (0.031) | -2.7% (0.146) | +0.5% (0.547) | -7.0% (0.011) |

A better daily fit did not turn into better intraday forecasts at 15 and 30 minutes. In the last hour, simply using VIX as the daily level helps.

The hybrid against these benchmarks (hybrid margin in % of the benchmark's QLIKE, one-sided DM p that the hybrid is better):

| Horizon | Subset | vs mcsGARCH | vs best matched mcsGARCH (which) | vs linear_base |
|---|---|---|---|---|
| next 15 | all | -5.8% (below 0.001) | -4.8% (0.005), mcsgarch_iv | -3.7% (0.016) |
| next 15 | expiry | -6.6% (0.001) | -6.0% (0.040), mcsgarchx_vix_iv | -5.7% (0.079) |
| next 15 | pre-auction | -5.5% (below 0.001) | -4.7% (below 0.001), mcsgarch_vix | -4.7% (0.008) |
| next 15 | hold-out | -6.4% (below 0.001) | -6.2% (0.003), mcsgarchx_vix_iv | -3.0% (0.112) |
| next 30 | all | -7.6% (below 0.001) | -5.9% (below 0.001), mcsgarch_iv | -3.9% (0.006) |
| next 30 | expiry | -6.2% (below 0.001) | -4.7% (0.098), mcsgarchx_vix_iv | -4.6% (0.084) |
| next 30 | pre-auction | -6.9% (below 0.001) | -5.2% (0.004), mcsgarch_iv | -4.0% (0.010) |
| next 30 | hold-out | -9.2% (below 0.001) | -8.3% (below 0.001), mcsgarchx_vix_iv | -4.0% (0.024) |
| last hour | all | -6.4% (below 0.001) | -1.7% (0.227), mcsgarch_vix | +0.6% (0.623) |
| last hour | expiry | -7.9% (0.010) | +1.0% (0.557), mcsgarch_vix | -1.6% (0.391) |
| last hour | pre-auction | -7.0% (0.002) | -0.9% (0.392), mcsgarch_vix | +0.7% (0.594) |
| last hour | hold-out | -7.1% (below 0.001) | +4.6% (0.849), mcsgarchx_vix_iv | +1.9% (0.816) |

**What it changes.**

* At 15 and 30 minutes, "beats the paper's model" survives against the best information-matched mcsGARCH on all days, before the auction and in the hold-out (margins 4.7% to 8.3%, p at most 0.005). On expiry days it is weaker (p 0.040 at 15 minutes, 0.098 at 30 minutes). The tree correction also beats a linear model on the same features at these horizons on all days (p 0.016 and 0.006).
* In the last hour it does **not** survive: mcsGARCH with VIX as its daily level is within 1.7% of the hybrid (p 0.227), and a ridge regression on the same features is as good as the hybrid. The last-hour gain over mcsGARCH is mostly the value of the VIX information, not of the hybrid method.
* The contribution should read "adding option-implied and VIX information and a tree-based correction to our implementation of mcsGARCH lowers QLIKE", with the 15 and 30 minute gain also holding against an mcsGARCH that is given that information. It should say that CPP made no forecasts, so this is not a criticism of CPP.

## M4.2 and minor comment 15. What the Model Confidence Set keeps

**What we did.** Read the committed MCS files (`results/pipeline_nse_v2/mcs_*.csv`: 10% level, daily QLIKE, 1,000 replications, block 5), recomputed them with the pipeline's settings (identical p-values, maximum difference 0.0), and again with 10,000 replications and blocks of 5 and 10 days.

**Result (REAL DATA; `results/referee_forecast/mcs_summary.csv`, `rank_by_mean_qlike.csv`; committed MCS files).** Committed settings:

| Horizon | Subset | Models kept | Top-ranked model | hybrid_base MCS p | Models with a higher MCS p than hybrid_base | hybrid_base rank by mean QLIKE | mcsGARCH (p) | HAR (p) | VIX (p) |
|---|---|---|---|---|---|---|---|---|---|
| next 15 | all | 18 of 32 | hybrid_gex_us | 0.902 | 1 | 3 of 32 | out (0.000) | out (0.000) | out (0.000) |
| next 15 | expiry | 25 of 32 | hybrid_gex_participant | 0.956 | 1 | 2 of 32 | out (0.046) | out (0.008) | out (0.000) |
| next 15 | pre-auction | 15 of 32 | hybrid_gex_participant | 0.489 | 10 | 6 of 32 | out (0.000) | out (0.000) | out (0.000) |
| next 30 | all | 18 of 32 | hybrid_v1gex | 0.595 | 14 | 7 of 32 | out (0.000) | out (0.000) | out (0.000) |
| next 30 | expiry | 25 of 32 | hybrid_gex_participant | 0.843 | 10 | 7 of 32 | out (0.024) | out (0.000) | out (0.000) |
| next 30 | pre-auction | 18 of 32 | hybrid_v1gex | 0.948 | 7 | 6 of 32 | out (0.000) | out (0.000) | out (0.000) |
| last hour | all | 27 of 32 | linear_v1 | 0.966 | 8 | 9 of 32 | out (0.021) | **in** (0.102) | **in** (0.229) |
| last hour | expiry | 28 of 32 | linear_v1 | 0.996 | 1 | 2 of 32 | **in** (0.439) | **in** (0.184) | **in** (0.416) |
| last hour | pre-auction | 24 of 32 | linear_gex_us | 0.936 | 1 | 8 of 32 | out (0.083) | out (0.084) | out (0.000) |

In the last hour on all days, VIX alone has QLIKE 20.4% above mcsGARCH and HAR 1.9% above (`scoreboard_last_hour.csv`), yet both stay in the set. With 10,000 replications the borderline cases move: HAR leaves the last-hour set with 5-day blocks (p 0.088) and stays with 10-day blocks (p 0.115); mcsGARCH stays in the last-hour expiry set in every setting (p 0.390 to 0.439).

**What it changes.** The paper should say plainly:

* at 15 and 30 minutes the MCS excludes mcsGARCH, HAR and VIX on all days, on expiry days and before the auction;
* in the last hour it has little power: one forecast per day, and it keeps VIX alone and HAR on all days and mcsGARCH on expiry days, so it does not support "also on expiry days" for the last hour;
* the top-ranked model is `hybrid_gex_us` at 15 minutes, `hybrid_v1gex` at 30 minutes and `linear_v1` in the last hour; `hybrid_base` is never top-ranked (MCS p 0.902, 0.595 and 0.966 on all days, ranked 3, 7 and 9 of 32 by mean QLIKE). At 15 and 30 minutes every model with a lower mean QLIKE is a hybrid with another feature set; in the last hour they are ridge models (`linear_*`, including `linear_base`) and tuned LightGBM (`scoreboard_*.csv`, group `all`). The MCS cannot separate any of them from `hybrid_base`.

## M4.4. The hold-out: plain LightGBM against the hybrid

**What the referee said.** In the hold-out plain LightGBM beats the hybrid at every horizon, so "a tree model needs the hybrid structure" is period-specific. More basically, the hybrid became the headline after the first full-sample run, so the hold-out cannot validate that choice.

**What we did.** Split the 792 test days into the development window (462 days, before 1 July 2025) and the hold-out (330 days, from 1 July 2025, of which 267 are before the closing auction began on 3 August 2026). For each we report each model against mcsGARCH, and a two-sided DM test of plain LightGBM (`lgbm_base`) and tuned LightGBM (`lgbmt_base`) against the hybrid.

**Result (REAL DATA; `results/referee_forecast/holdout_table.csv`, `holdout_lightgbm_vs_hybrid.csv`).** Margin against mcsGARCH (one-sided p that the model beats mcsGARCH):

| Horizon | Window | lgbm_base | lgbmt_base | hybrid_base | linear_base |
|---|---|---|---|---|---|
| next 15 | development | -0.4% (0.459) | -2.7% (0.041) | -5.3% (below 0.001) | -1.2% (0.304) |
| next 15 | hold-out | -7.7% (0.001) | -8.1% (below 0.001) | -6.4% (below 0.001) | -3.5% (0.087) |
| next 30 | development | +1.1% (0.562) | -3.0% (0.184) | -6.6% (below 0.001) | -2.9% (0.116) |
| next 30 | hold-out | -10.4% (below 0.001) | -10.5% (below 0.001) | -9.2% (below 0.001) | -5.5% (0.009) |
| last hour | development | +3.7% (0.670) | +0.9% (0.552) | -5.3% (0.048) | -4.2% (0.241) |
| last hour | hold-out | -12.9% (0.007) | -12.6% (0.006) | -7.1% (below 0.001) | -8.8% (0.001) |

LightGBM against the hybrid (difference in QLIKE, negative means LightGBM is better; two-sided p):

| Horizon | Window | lgbm_base | lgbmt_base |
|---|---|---|---|
| next 15 | development | +5.2% (0.202) | +2.8% (0.047) |
| next 15 | hold-out | -1.4% (0.065) | -1.8% (0.022) |
| next 15 | hold-out before the auction | -0.1% (0.891) | -0.5% (0.412) |
| next 15 | all 792 days | +2.3% (0.323) | +0.8% (0.384) |
| next 30 | development | +8.2% (0.301) | +3.9% (0.337) |
| next 30 | hold-out | -1.3% (0.180) | -1.4% (0.153) |
| next 30 | hold-out before the auction | -0.4% (0.556) | -0.6% (0.498) |
| next 30 | all 792 days | +4.5% (0.354) | +1.8% (0.463) |
| last hour | development | +9.4% (0.206) | +6.5% (0.250) |
| last hour | hold-out | -6.2% (0.171) | -6.0% (0.146) |
| last hour | hold-out before the auction | +0.4% (0.848) | -2.0% (0.476) |
| last hour | all 792 days | +0.2% (0.969) | -0.9% (0.792) |

**What it changes.**

* LightGBM is ahead of the hybrid in the hold-out and behind it in the development window, and the only significant differences at 5% are tuned LightGBM at 15 minutes, in opposite directions (hybrid ahead in development, p 0.047; LightGBM ahead in the hold-out, p 0.022). Over all 792 days the two are not distinguishable (p 0.32 to 0.97).
* Before the auction, LightGBM and the hybrid are almost equal in the hold-out (differences of 2.0% or less, p at least 0.41). LightGBM's hold-out lead comes from the 63 days after the auction began (330 minus 267).
* So the claim that "a tree model beats CPP's model only with the hybrid structure" is not supported and should be dropped. What is supported: in the development window plain LightGBM does not beat mcsGARCH, in the hold-out it does, and the hybrid beats mcsGARCH in both.
* The hold-out checks the later tuning choices of the final protocol, not the choice of the hybrid as the headline model (the hybrid was chosen after the first full-sample run). The words "evaluated once" should go.

## M5. The ranking under MSE

From the committed scoreboards, tests and MCS files (`results/referee_forecast/mse_ranking.csv`, built from `results/pipeline_nse_v2/scoreboard_*.csv`, `tests_*.csv`, `mcs_*_mse.csv`; and `results/forecast_diagnostics/largest_mcsgarch_errors.csv`). Under mean squared error the ranking at 15 and 30 minutes nearly reverses: mcsGARCH ranks 31 of 32, HAR, VIX and plain LightGBM have 84.5% to 86.0% lower MSE, ridge (`linear_base`) ranks 3rd, and the hybrid has only 33.2% and 26.9% lower MSE (ranks 24 and 30). The reason is a handful of windows. The ten largest squared errors of mcsGARCH are 96.8% of its total at 15 minutes, 95.9% at 30 minutes and 62.0% in the last hour, led by 09:15 on 5 June 2024, the morning after the election-result crash, when mcsGARCH forecast a 15-minute variance of 0.00201 against a realized 0.0000986; the hybrid, which multiplies mcsGARCH by a correction, forecast 0.00163 and inherits the error. QLIKE punishes a forecast that is too high only mildly and counts every window in proportion, so these few windows barely matter for it; MSE is dominated by them. As a result MSE cannot rank the models here: the MSE Model Confidence Set keeps 31 of 32 models at every horizon (only VIX is excluded), no MSE difference against mcsGARCH is significant at 15 or 30 minutes (smallest p 0.131), and in the last hour the smallest p is 0.023 for one of 31 comparisons (`hybridiv_gex_placebo`), with the hybrid at 0.053. We state the conclusions for QLIKE, the loss that ranks variance forecasts consistently with a noisy proxy (Patton 2011), and say that a user whose costs grow with the square of the error, such as a hedger hurt mostly by the rare crash morning, should not read the QLIKE ranking as the answer to their question. We offer no economic evaluation (VaR, hedging); that remains a limitation.

## Side findings from rebuilding the benchmark

**1. The EWMA fallback in the benchmark used later data.** The pipeline replaces the daily EGARCH by an EWMA when the EGARCH variance on any day of the sample is 5 log points or more away from the training variance. That check covers days after the test block too, so it is not a real-time rule. In the refit starting at day 192, the EGARCH failed the check only because of day 562 (21 April 2025), long after its test block; in the refit starting at day 171, the first bad day was day 189 (9 October 2023), inside its own block (`mcsgarchx_params.csv`). A strictly real-time version (`mcsgarch_realtime`: keep the EGARCH until its first bad test day) is worse than the committed mcsGARCH at 15 and 30 minutes (+5.5% and +6.9%) and the same in the last hour (-0.1%). So the look-ahead made the benchmark stronger, not weaker. The hybrid also uses this mcsGARCH as its starting point, so the clean check is to leave out the 42 test days of those two refits: the hybrid's margin over mcsGARCH is then 5.9%, 7.9% and 6.5% (all p below 0.001; `fair_benchmarks.csv`, subset `no_ewma_fallback`), against 5.8%, 7.6% and 6.4% on all days. The result does not depend on it, but the paper should disclose the rule and the pipeline should use the real-time version.

**2. The first refit is sensitive to the starting value.** The arch package starts the daily EGARCH recursion from a backcast computed from a regression on the whole series. In the first refit (training days 0 to 149, test days 150 to 170), recomputing the paper's daily variance with a backcast from training days only changes it by up to 98.8% on the test days; in every other refit the difference is at rounding level (median 6.7e-16; `mcsgarchx_params.csv`, column `paper_h_backcast_max_rel_diff_test_days`). This affects one block of 21 test days, fitted on only 150 days. The new mcsGARCH-X variants use the training-days backcast throughout.

## Referee priority 5. Giacomini and White tests: can the hybrid's advantage be predicted from conditions?

**POST HOC, REAL DATA, added after the first version of this note.** Code: `experiments/referee_forecast_gw.py`; log: `results/referee_forecast_gw_stdout.txt`; files: `results/referee_forecast/gw_tests.csv`, `gw_regression_coefficients.csv`, `gw_group_margins.csv`; tests on synthetic data: `tests/test_referee_forecast_gw.py`.

**What we did.** The DM test asks whether the hybrid is better *on average*. Giacomini and White (2006) also ask a sharper question: could you have predicted, the evening before, on which days the hybrid would do better? For `hybrid_base` against mcsGARCH and against the best information-matched mcsGARCH of each horizon (`mcsgarch_iv` at 15 and 30 minutes, `mcsgarch_vix` in the last hour, the ones with the lowest loss on all days in `fair_benchmarks.csv`), we took the daily loss difference d_t = benchmark QLIKE minus hybrid QLIKE (positive: hybrid better), in points of the benchmark's mean daily QLIKE, on 791 days (the first test day has no previous day). The tests:

* **Unconditional** (instrument: a constant): is the average advantage zero? Two-sided, Newey-West variance. This is the square of the DM statistic.
* **Conditional C1** (a constant and the previous day's d) and **C2** (C1 plus an expiry-day dummy and dummies for the Thursday-weekly and Tuesday-weekly regimes), with Giacomini and White's own variance for one-step forecasts, (1/n) times the sum of Z_t Z_t'; a Newey-West version is a robustness check.
* **Slope tests.** A conditional test rejects whenever the average advantage is not zero, even if nothing predicts it. To ask only "does the advantage vary with conditions?", we regress d_t on the instruments and test that the slopes are zero (all of them, the lag alone, the expiry dummy alone, the two regime dummies), with a Newey-West covariance.

Every instrument is known before day t: the previous day's loss at its close, the expiry calendar and the regime in advance. The regimes are consecutive periods: multi-weekly expiries to 19 November 2024, a single Thursday weekly to 29 August 2025, a single Tuesday weekly from 1 September 2025. The same caveat as in M4.3 applies: Giacomini and White's theory assumes a rolling estimation window, and ours is expanding, so we use these as tests of conditional moments of the realized loss differences.

**Counting the p-values.** 6 comparisons (3 horizons times 2 benchmarks) times 7 tests give 42 p-values in the main family, Holm-adjusted over all 42; the 12 Newey-West versions of C1 and C2 are a robustness check; 54 in all. The synthetic tests check that every test rejects about 5% of the time when the two forecasts are equally good (heavy-tailed differences, the real sample size and instruments), that the slope tests keep that size when one model is better by a constant amount with autocorrelated differences, and that the conditional test finds an advantage that exists only on expiry days, which the unconditional test cannot see.

**Result (REAL DATA; `gw_tests.csv`).** p-value, with the Holm-adjusted p over the 42 in parentheses:

| Horizon | Benchmark | Mean d (points) | Unconditional | C1 | C2 | All slopes | Lag | Expiry | Regimes |
|---|---|---|---|---|---|---|---|---|---|
| next 15 | mcsGARCH | +5.82 | below 0.001 (below 0.001) | below 0.001 (below 0.001) | below 0.001 (below 0.001) | 0.111 (1.000) | 0.046 (1.000) | 0.251 (1.000) | 0.425 (1.000) |
| next 15 | mcsgarch_iv | +4.83 | 0.010 (0.292) | 0.016 (0.441) | 0.018 (0.467) | 0.059 (1.000) | 0.151 (1.000) | 0.234 (1.000) | 0.100 (1.000) |
| next 30 | mcsGARCH | +7.67 | below 0.001 (below 0.001) | below 0.001 (below 0.001) | below 0.001 (below 0.001) | 0.849 (1.000) | 0.376 (1.000) | 0.883 (1.000) | 0.530 (1.000) |
| next 30 | mcsgarch_iv | +5.89 | 0.001 (0.038) | 0.001 (0.035) | below 0.001 (0.014) | 0.004 (0.126) | 0.115 (1.000) | 0.605 (1.000) | 0.027 (0.652) |
| last hour | mcsGARCH | +6.32 | below 0.001 (0.008) | below 0.001 (0.022) | 0.005 (0.158) | 0.023 (0.573) | 0.908 (1.000) | 0.652 (1.000) | 0.005 (0.139) |
| last hour | mcsgarch_vix | +1.66 | 0.455 (1.000) | 0.750 (1.000) | 0.408 (1.000) | 0.567 (1.000) | 0.767 (1.000) | 0.612 (1.000) | 0.232 (1.000) |

The Newey-West versions of C1 and C2 are close to these, with one exception: C2 against `mcsgarch_iv` at 15 minutes gives p 0.098 instead of 0.018, which again makes the 15 minute edge over the matched benchmark the least secure. Of the 42 p-values, 20 are below 0.05 before adjustment and 11 after.

The hybrid's margin within each group (% of the benchmark's QLIKE in that group; descriptive, `gw_group_margins.csv`):

| Horizon | Benchmark | Expiry days | Other days | Multi-weekly | Thursday weekly | Tuesday weekly |
|---|---|---|---|---|---|---|
| next 15 | mcsGARCH | -6.6% | -5.5% | -5.2% | -5.2% | -6.8% |
| next 15 | mcsgarch_iv | -8.0% | -3.7% | -0.2% | -7.3% | -7.0% |
| next 30 | mcsGARCH | -6.2% | -8.1% | -6.7% | -6.4% | -9.7% |
| next 30 | mcsgarch_iv | -6.4% | -5.7% | -0.7% | -8.2% | -9.3% |
| last hour | mcsGARCH | -7.9% | -5.9% | -9.9% | +0.0% | -7.1% |
| last hour | mcsgarch_vix | +1.0% | -2.3% | +1.7% | +0.9% | -4.0% |

(169 expiry days and 623 other days; 312, 192 and 288 days in the three regimes.)

**Reading.**

* **Expiry days: no.** None of the six expiry slope tests is significant (p 0.234 to 0.883), and the point estimates go both ways. The hybrid's advantage does not concentrate on expiry days.
* **Yesterday's advantage barely predicts today's.** Only one lag test is below 0.05 (15 minutes against mcsGARCH, p 0.046), and not after adjustment.
* **Time, or regime: possibly.** Against the information-matched mcsGARCH at 15 and 30 minutes, the hybrid's edge is close to zero in the multi-weekly period (-0.2% and -0.7%; +0.1% and -0.6% if the 42 days of the EWMA-fallback refits are left out) and 7.0% to 9.3% in the two weekly periods since November 2024. The regime test gives p 0.100 and 0.027, not significant after adjustment (0.652 at 30 minutes). In the last hour against mcsGARCH the gain is 9.9% in the multi-weekly period, 0.0% in the Thursday period and 7.1% in the Tuesday period (p 0.005, adjusted 0.139). Because each regime is one stretch of time, these tests cannot separate an expiry-calendar effect from a time effect: LightGBM is trained on a longer history in later periods, and the market changed. The development and hold-out comparison against `mcsgarch_iv` (development -3.1% and -3.5%, p 0.097 and 0.066; `fair_benchmarks.csv`) points the same way.
* **Average advantage.** The unconditional tests agree with the DM tests. After the adjustment over all 42 p-values, the advantage survives against mcsGARCH at all three horizons (in the last hour for the unconditional test and C1) and against `mcsgarch_iv` at 30 minutes (adjusted p 0.014 to 0.038). It does not survive against `mcsgarch_iv` at 15 minutes (adjusted p 0.292 to 0.467, before adjustment 0.010 to 0.018). That family includes 24 slope tests, so this is a strict standard. In the last hour nothing separates the hybrid from `mcsgarch_vix`.

**What it changes.** Nothing about expiry days: the advantage is neither concentrated there nor absent there. One new qualification: the 15 and 30 minute edge over the information-matched mcsGARCH comes from the period since November 2024, and the 15 minute edge is the less secure of the two. The rewritten claim below adds one sentence for this.

## What changes in the claims

1. **Abstract and contribution 2.** Replace "CPP's model can be beaten out of sample at every horizon" by a claim about adding information and a tree-based correction to our implementation of mcsGARCH, and say that at 15 and 30 minutes the gain also holds against an mcsGARCH given the same information, while in the last hour it does not. Say that CPP made no forecasts.
2. **"Also on expiry days".** Keep it for 15 and 30 minutes; drop it for the last hour (step-down p 0.062; the MCS keeps mcsGARCH, p 0.439).
3. **"Also on a hold-out evaluated once".** Drop "evaluated once"; say the hold-out checks the later tuning of the protocol, not the choice of the hybrid.
4. **"The only one with p at most 0.010 in all six columns".** Replace by the SPA and step-down results.
5. **"A tree model beats CPP's model only with the hybrid structure".** Drop: plain LightGBM and the hybrid are not distinguishable over all days.
6. **Section 6.2 (MCS).** Add the top-ranked models, the MCS p of `hybrid_base`, and the last-hour set (VIX and HAR on all days, mcsGARCH on expiry days).
7. **DM p-values.** Quote noise-adjusted p-values or "below 0.001", and say that a full bootstrap over seeds was not run (cost at most 6.3 hours).
8. **Appendix D.5.** Disclose that the EWMA fallback rule looked at later days, and give the real-time check.
9. **Giacomini and White (priority 5).** Say that the hybrid's advantage does not concentrate on expiry days, and that its 15 and 30 minute edge over the information-matched mcsGARCH comes from the weekly-expiry periods since November 2024 (descriptive; the regime test is not significant after adjustment).
10. Unchanged: GEX features add nothing; the 15 and 30 minute gains over mcsGARCH survive every check here.

## Rewritten forecasting claim

> Out of sample, over 792 test days, adding the previous evening's option chain, India VIX and calendar features to our implementation of the mcsGARCH model of Chhimwal, Pandey and Pandey (who fitted it but made no forecasts), through a LightGBM correction of its errors, lowers QLIKE loss by 5.8% and 7.6% for the next 15 and 30 minutes (one-sided Diebold-Mariano p below 0.001). In checks made after review, these gains survive a correction for having compared 31 challengers (Hansen's SPA and step-down adjusted p below 0.001), plausible seed noise (it would have to be at least 3.4 and 4.5 points, against 0.19 measured), the exclusion of the 42 days on which the benchmark used a fallback chosen with later data, and, most importantly, a benchmark given the same information: against the best of four mcsGARCH variants that use VIX or the straddle-implied variance, the hybrid is 4.8% (p 0.005) and 5.9% (p below 0.001) better, and against a ridge regression on the same features 3.7% and 3.9% better. Giacomini and White tests find that this edge over the matched benchmark does not concentrate on expiry days, but it comes from the weekly-expiry periods since November 2024 (in the multi-weekly period before, it is 0.2% and 0.7%), and at 15 minutes it does not survive an adjustment over all 42 of those tests. On expiry days the gain over the best matched benchmark is weaker (p 0.040 and 0.098). In the last hour the 6.4% gain over mcsGARCH is mostly information rather than method: mcsGARCH with India VIX as its daily level comes within 1.7% of the hybrid (p 0.227), and on expiry days no model beats mcsGARCH after the multiplicity correction. The hybrid structure itself is not shown to matter: plain LightGBM cannot be distinguished from it over the full sample (p 0.32 to 0.97) and is ahead in the hold-out, which checks later tuning choices rather than the choice of the hybrid. Under squared-error loss the models cannot be ranked (the MSE Model Confidence Set keeps 31 of 32). Dealer-gamma features add nothing.

## Files and reproduction

New files (nothing in the pipeline or in `results/pipeline_nse_v2/` was changed):

* `experiments/referee_forecast.py`: all seven sections.
* `results/referee_forecast/`: `spa_reality_check.csv`, `spa_adjusted_pvalues_by_model.csv`, `mcsgarchx_forecasts_next15.csv`, `mcsgarchx_forecasts_next30.csv`, `mcsgarchx_forecasts_last_hour.csv`, `mcsgarchx_params.csv`, `mcsgarchx_run_info.json`, `fair_benchmarks.csv`, `seed_noise_summary.csv`, `seed_adjusted_dm.csv`, `refit_envelope.csv`, `seed_rerun_cost.csv`, `mcs_summary.csv`, `rank_by_mean_qlike.csv`, `holdout_table.csv`, `holdout_lightgbm_vs_hybrid.csv`, `mse_ranking.csv`.
* `results/referee_forecast_stdout.txt`: the printed log of the run (193 seconds).
* `experiments/referee_forecast_gw.py`, `results/referee_forecast/gw_tests.csv`, `gw_regression_coefficients.csv`, `gw_group_margins.csv` and `results/referee_forecast_gw_stdout.txt`: the Giacomini and White tests (priority 5). It reads the committed forecasts and the cached mcsGARCH-X forecasts and runs in a few seconds: `PYTHONPATH=src python experiments/referee_forecast_gw.py > results/referee_forecast_gw_stdout.txt`.
* `tests/test_referee_forecast.py`, `tests/test_referee_forecast_gw.py`: synthetic data only.
* `docs/REFEREE_RESPONSE_FORECAST.md`: this note.

To reproduce:

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/referee_forecast.py > results/referee_forecast_stdout.txt

The mcsGARCH-X forecasts are cached in `results/referee_forecast/`; `--rebuild-mcsgarchx` refits them and needs the raw NSE files in `data/raw`. Use `OMP_NUM_THREADS=1`: with several threads, one arch EGARCH fit in the rebuild converged differently in our trials.

## References

* Chhimwal, Pandey and Pandey (2025), as cited in the working paper (CPP).
* Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business and Economic Statistics* 13(3), 253 to 263.
* Engle, R. F. and Sokalska, M. E. (2012). Forecasting intraday volatility in the US equity market. Multiplicative component GARCH. *Journal of Financial Econometrics* 10(1), 54 to 83.
* Giacomini, R. and White, H. (2006). Tests of conditional predictive ability. *Econometrica* 74(6), 1545 to 1578.
* Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business and Economic Statistics* 23(4), 365 to 380.
* Hansen, P. R., Lunde, A. and Nason, J. M. (2011). The model confidence set. *Econometrica* 79(2), 453 to 497.
* Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting* 13(2), 281 to 291.
* Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: a new approach. *Econometrica* 59(2), 347 to 370.
* Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics* 160(1), 246 to 256.
* Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association* 89(428), 1303 to 1313.
* Romano, J. P. and Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica* 73(4), 1237 to 1282.
* White, H. (2000). A reality check for data snooping. *Econometrica* 68(5), 1097 to 1126.
