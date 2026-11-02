# Iteration log

A record of what was tried, what went wrong, and what changed as a result. Entries are in the order they happened during the build on 2 November 2026. "Synthetic" means simulated data; nothing here uses real market data.

## Iteration 1: reading before building

* Read the full paper, the earlier reading notes, the paper summary and the component plan.
* Decision: the headline is an out-of-sample forecasting contest (positioning features versus the paper's mcsGARCH), not an in-sample regression, because (a) the paper makes no forecasts, (b) a forecast contest gives a result even if the dealer sign cannot be identified, and (c) it uses the paper's own model as the benchmark.
* Checked network access: NSE, BSE, Yahoo and broker APIs are blocked from the build machine. Decision: build and validate everything on a simulator with known parameters, and build the real-data path so it runs unchanged once data are downloaded elsewhere.

## Iteration 2: engines and their tests

* Black-76 greeks, IV solver, GEX, realized measures, mcsGARCH, Diebold-Yilmaz, expiry calendar.
* **Test failure 1 (real finding about a measure).** The Rogers-Satchell test failed: on Brownian bars monitored at 500 points per bar, RS underestimated the true variance by about 8%. Cause: a bar's observed high and low understate the true path's extremes when the path is only sampled at discrete points. This is a known property of range estimators, not a code bug. Fix: the test now monitors at 20,000 points (bias about 1%). Consequence for the project: the simulator draws only 8 intrabar points, so its RS values are biased low. That does not affect the forecasting experiments (targets are sums of squared returns) or the before/after experiment E2 (both periods share the same bias), but RS levels from the simulator should not be compared with real RS levels.
* **Test failure 2 (test, not code).** The GEX finite-difference check failed at a relative error of 2.7e-4 because the finite-difference step (0.1% of spot) was too large. Fix: smaller step (0.01%), and the identity then holds to 1e-4.
* Renamed `calendar.py` to `expiry_calendar.py` so it can never shadow Python's standard `calendar` module.

## Iteration 3: simulator timing bug

* First simulator version built each day's option chain from that day's opening spot and published it at the same day's close, so the "published" chain did not match the chain that was live the next day.
* Fix: at the close of day t the simulator now creates the chain that will be live on day t+1 (open interest is fixed overnight), publishes it as the bhavcopy, and publishes the participant file and VIX describing day t+1. Dealer positions during day t+1 equal the positions at the close of t. This matches how the real files are timed and removes any look-ahead.

## Iteration 4: mcsGARCH validation

* On 250 synthetic days with true $(\alpha, \beta, \nu) = (0.03, 0.95, 7)$, the estimates were (0.031, 0.949, 7.2). Fit time 0.4 s for 94,000 minutes.
* The raw diurnal curve is noisy at the minute level (largest single-minute error about 55% with 250 days; shape correlation 0.91). Added optional moving-average smoothing.
* **Unexpected observation that led to experiment E5.** Fitting ARMA(1,1) to synthetic returns that have no autocorrelation gave $\phi = 0.84$, $\theta = -0.84$. The paper's Table 2 shows the same pattern in all four columns. E5 then showed this is what ARMA(1,1) does on near-white-noise data.

## Iteration 5: first end-to-end forecast run

* Synthetic market, 300 days, gamma effect on (kappa 0.4), walk-forward with monthly refits.
* Results (synthetic): the hybrid model beat mcsGARCH by about 5% QLIKE; LightGBM and HAR were close behind; the GEX ablation showed **no** gain.
* Diagnosis: dealer positioning in the simulator is persistent (AR coefficient 0.9), so the volatility it causes shows up in recent realized volatility, which the base model already sees. Positioning adds information only through its changes and through its interaction with time to expiry. This is likely true of real markets too, and it is an important design lesson: a GEX test needs a strong baseline, or it will "find" information that recent volatility already carries.
* Changes: added an oracle GEX feature (true dealer positions, simulator only) as an upper bound; split GEX features into sets by sign rule so each can be ablated separately; designed the power study (E4).
* The last-hour target has one observation per day; with about 250 training days LightGBM overfits and the hybrid was worse than mcsGARCH on expiry days in some runs. Kept as a finding: with one observation per day, simple models are favoured.

## Iteration 6: speed

* The daily GEX panel took 88 s for 300 days because the gamma-flip root search re-created a pandas table for every grid point. Rewrote it to evaluate the whole grid with one numpy broadcast. Now about 4 s per 100 days, with unchanged results (tests pass).
* The first power-study configuration (4 effect sizes x 2 noise levels x 10 replications, monthly refits) ran at about 5 minutes per market and would have taken hours. Changed to two-month test blocks and a 3 x 2 x 8 grid. Also found the numerical libraries were starting extra threads in each worker (load average twice the core count); set single-threaded BLAS in the experiment scripts.

## Iteration 7: real-data path

* Built `build_market_from_nse` (bhavcopy in both NSE layouts, participant-wise OI, 1-minute bars, VIX) and `export_nse_layout`, which writes a simulated market to files in NSE's layouts.
* **Bug found by the round-trip test:** the participant-OI reader failed on the file's one-line title because pandas could not tokenise a ragged first line. Fixed by locating the header row with a plain text scan.
* The round trip now reproduces returns, VIX, OI (converted from units to contracts) and participant signs exactly.

## Iteration 8: the power study caught two problems

The first power-study pass was stopped after 13 markets because the results looked wrong in two ways.

* **Daily EGARCH divergence (a real robustness bug).** In two markets the mcsGARCH forecasts were off by a factor of about 10^8. Tracing one case: the `arch` AR(1)-EGARCH(1,1) fit on a 402-day training window converged to nonsense (AR coefficient 14.7, asymmetry -874, persistence 1.0), and every later forecast inherited it. Even the "good" fits on 150 to 340 days had a flat likelihood (EGARCH persistence near zero, daily variance between 0.45 and 2.3 times the truth). Fix: `daily_egarch_variance` now checks the fit (finite parameters, stationary AR term, persistence in [0, 1), filtered variance within a factor of e^5 of the sample variance), retries with normal errors, and falls back to an EWMA variance if both fail. The method used is reported. On the failing case the retry with normal errors succeeds. Lesson for the real data: the daily component deserves its own check, and the paper's choice of EGARCH for $h_t$ should be compared with a simpler EWMA or HAR daily forecast.
* **Simulator leak through option implied vols.** With the gamma effect switched off, GEX features still improved forecasts by up to 5% in some cells. Cause: the simulator set each chain's implied vol from the *true* next-day variance, while the published VIX carried 5% noise. Gross GEX depends on implied vol, so it gave the model a cleaner copy of next-day variance than VIX did. Real implied vols are noisy expectations, not the truth. Fix: the chain's ATM implied vol now equals the published (noisy) VIX value, so chain features carry no information about next-day variance beyond VIX. This is exactly the kind of false positive the zero-effect cells exist to catch.
* Both fixes invalidate the first 13 markets, so the power study was restarted from scratch. The discarded partial results are not reported anywhere.

## Iteration 9: the before/after experiment needed a better null

* The first version of E2 generated the "before" and "after" windows as two independent draws of a process whose daily volatility level is persistent (AR(1) coefficient 0.9). Under that null, even day-level tests rejected 55 to 65% of the time, not 5%.
* That is not a bug in the tests. With persistent volatility, two separate three-month windows genuinely differ in their average volatility level, so "volatility differs between the windows" is true even when nothing about expiry changed. A day-level test correctly detects the difference; it just answers a different question from the one the paper asks.
* So E2 now has two scenarios. Scenario A makes days independent, which isolates the within-day dependence problem (the one the minute-level Wilcoxon test has). Scenario B keeps persistent daily volatility and adds a difference-in-differences test: compare each expiry day with the other days of its own week, then compare those weekly differences before and after. Persistent shocks common to the whole week cancel.
* Design lesson for the real study: a before/after comparison of expiry-day volatility should be measured relative to non-expiry days in the same week (or with a regression that includes week effects), not as raw levels.

## Iteration 10: a new test caught a zero-variance fallback

* Writing a test for the EWMA fallback (feeding the daily model a degenerate series) showed the fallback could return a daily variance of exactly zero, which would make every later normalisation divide by zero.
* Fix: the EWMA now starts from the training window's mean square and is floored at 1% of the overall mean square. The fallback has not been triggered in any experiment run so far (E1a: 0 of 180 fits), but it is now safe if it is.
