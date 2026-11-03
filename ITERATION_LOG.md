# Iteration log

A record of what was tried, what went wrong, and what changed as a result, in the order it happened (2 and 3 November 2026). Each iteration uses the same ten fields. Iterations 1 to 11 used only **synthetic** (simulated) data. From iteration 12 on, results are **REAL DATA** (NSE files plus validated third-party 1-minute bars) unless marked otherwise.

Short glossary: **GEX** is dealer gamma exposure (how much hedging trading dealers must do per 1% index move); **mcsGARCH** is the paper's intraday volatility model; **QLIKE** is a standard loss for volatility forecasts (lower is better); **bhavcopy** is NSE's end-of-day file of every derivative contract.

---

## Iteration 1: reading before building

* **OBJECTIVE:** Decide what the prototype should test before writing code.
* **PROBLEM IDENTIFIED:** The paper (Chhimwal, Pandey and Pandey 2025) is a before/after study of one expiry rule change; it makes no forecasts and does not use dealer positioning. A prototype that only re-runs its tables would add little.
* **HYPOTHESIS:** An out-of-sample forecasting contest (dealer-positioning features against the paper's own mcsGARCH) gives a result whether or not the dealer sign can be identified, and uses the paper's model as the benchmark.
* **CHANGES IMPLEMENTED:** Chose the forecasting contest as the headline, with the paper's tables as a replication track.
* **EXPERIMENTS PERFORMED:** Checked network access from the build machine.
* **RESULTS:** NSE, BSE, Yahoo and broker APIs were blocked at that point.
* **IMPROVEMENTS:** Built every engine to run first on a simulator with known parameters, and the real-data path to run unchanged once data could be reached.
* **REMAINING ISSUES:** No real data yet.
* **NEXT ITERATION:** Build and test the engines.

## Iteration 2: engines and their tests

* **OBJECTIVE:** Black-76 greeks, implied-vol solver, GEX, realized measures, mcsGARCH, Diebold-Yilmaz connectedness, expiry calendar, each with unit tests.
* **PROBLEM IDENTIFIED:** Two tests failed. (1) Rogers-Satchell variance on simulated Brownian bars sampled at 500 points per bar was about 8% too low. (2) The GEX finite-difference check failed at a relative error of 2.7e-4.
* **HYPOTHESIS:** (1) is a known property of range estimators (discretely sampled highs and lows understate the true range), not a code bug. (2) is a step size that is too large.
* **CHANGES IMPLEMENTED:** RS test now samples at 20,000 points (bias about 1%). GEX check uses a 0.01% step. Renamed `calendar.py` to `expiry_calendar.py` so it cannot shadow Python's `calendar` module.
* **EXPERIMENTS PERFORMED:** Reran the test suite.
* **RESULTS:** All tests pass; the GEX identity holds to 1e-4.
* **IMPROVEMENTS:** Documented that simulator RS levels are biased low (8 intrabar points) and must not be compared with real RS levels; the forecasting targets (sums of squared returns) are unaffected.
* **REMAINING ISSUES:** None for the engines.
* **NEXT ITERATION:** Make the simulator's timing match the real files.

## Iteration 3: simulator timing bug

* **OBJECTIVE:** Make simulated files appear in the same order as real NSE files.
* **PROBLEM IDENTIFIED:** The first simulator built each day's option chain from that day's opening price but published it at the same day's close, so the published chain did not match the chain live the next day.
* **HYPOTHESIS:** This creates a subtle look-ahead and a mismatch with how bhavcopies work.
* **CHANGES IMPLEMENTED:** At the close of day t the simulator now creates the chain live on day t+1 (open interest fixed overnight) and publishes it, the participant file and VIX for day t+1.
* **EXPERIMENTS PERFORMED:** Tests on dealer positions during day t+1 versus positions at the close of t.
* **RESULTS:** They are equal by construction.
* **IMPROVEMENTS:** No look-ahead in the synthetic experiments.
* **REMAINING ISSUES:** None.
* **NEXT ITERATION:** Validate mcsGARCH.

## Iteration 4: mcsGARCH validation

* **OBJECTIVE:** Check that the mcsGARCH implementation recovers known parameters.
* **PROBLEM IDENTIFIED:** The raw time-of-day (diurnal) curve is noisy at the minute level, and ARMA(1,1) fitted to returns with no autocorrelation gave phi = 0.84 and theta = -0.84.
* **HYPOTHESIS:** The diurnal noise needs smoothing; the ARMA pattern is what ARMA(1,1) does on near-white-noise data (the two roots cancel).
* **CHANGES IMPLEMENTED:** Optional moving-average smoothing of the diurnal curve. Added experiment E5 to study the ARMA pattern.
* **EXPERIMENTS PERFORMED:** 250 synthetic days with true (alpha, beta, nu) = (0.03, 0.95, 7).
* **RESULTS:** Estimates (0.031, 0.949, 7.2), fit time 0.4 s for 94,000 minutes. Diurnal shape correlation 0.91, largest single-minute error about 55%. E5 confirmed the cancelling-roots explanation; the paper's Table 2 shows the same pattern in all four columns.
* **IMPROVEMENTS:** A validated mcsGARCH and a concrete question about the paper's Table 2.
* **REMAINING ISSUES:** None for the model.
* **NEXT ITERATION:** First end-to-end forecasting run.

## Iteration 5: first end-to-end forecast run (synthetic)

* **OBJECTIVE:** Run the whole forecasting contest once.
* **PROBLEM IDENTIFIED:** With the simulated gamma effect switched on, GEX features still showed **no** gain. The last-hour target (one observation per day) made LightGBM overfit.
* **HYPOTHESIS:** Simulated dealer positions are persistent (AR coefficient 0.9), so their effect is already visible in recent realized volatility; positioning adds information only through its changes and its interaction with time to expiry.
* **CHANGES IMPLEMENTED:** Added an oracle GEX feature (true positions, simulator only) as an upper bound; split GEX features by sign rule; designed the power study (E4).
* **EXPERIMENTS PERFORMED:** 300 synthetic days, kappa 0.4, monthly walk-forward refits.
* **RESULTS:** Hybrid beat mcsGARCH by about 5% QLIKE; the GEX ablation showed no gain.
* **IMPROVEMENTS:** Design lesson kept for real data: a GEX test needs a strong baseline, or it will "find" information that recent volatility already carries.
* **REMAINING ISSUES:** Small samples favour simple models for the last hour.
* **NEXT ITERATION:** Speed, so the power study is feasible.

## Iteration 6: speed

* **OBJECTIVE:** Make the GEX panel and the power study fast enough to run many markets.
* **PROBLEM IDENTIFIED:** The daily GEX panel took 88 s for 300 days (a pandas table rebuilt for every grid point in the gamma-flip search). The first power-study grid would have taken hours, and numerical libraries started extra threads in each worker.
* **HYPOTHESIS:** Vectorising the flip search and pinning BLAS to one thread per worker will fix both.
* **CHANGES IMPLEMENTED:** One numpy broadcast for the flip grid; two-month test blocks; a 3 x 2 x 8 grid; single-threaded BLAS in experiment scripts.
* **EXPERIMENTS PERFORMED:** Timed the panel and reran the tests.
* **RESULTS:** About 4 s per 100 days, identical results.
* **IMPROVEMENTS:** The power study became feasible.
* **REMAINING ISSUES:** None.
* **NEXT ITERATION:** Build the real-data path.

## Iteration 7: real-data path

* **OBJECTIVE:** Read NSE's bhavcopy (both layouts), participant-wise OI, 1-minute bars and VIX.
* **PROBLEM IDENTIFIED:** A round-trip test (simulate, write in NSE layout, read back) failed: the participant-OI reader could not parse the file's one-line title.
* **HYPOTHESIS:** pandas cannot tokenise the ragged first line.
* **CHANGES IMPLEMENTED:** Locate the header row with a plain text scan. Added `export_nse_layout` for round trips.
* **EXPERIMENTS PERFORMED:** Round-trip test.
* **RESULTS:** Returns, VIX, OI and participant signs are reproduced exactly.
* **IMPROVEMENTS:** The real-data loader was tested before any real file was available.
* **REMAINING ISSUES:** Real files might still differ from the documented layouts (they did; see iteration 12).
* **NEXT ITERATION:** Power study.

## Iteration 8: the power study caught two problems

* **OBJECTIVE:** Measure how large a gamma effect the contest could detect (E4).
* **PROBLEM IDENTIFIED:** (1) In two markets mcsGARCH forecasts were off by about 10^8. (2) With the gamma effect switched off, GEX features still improved forecasts by up to 5% in some cells.
* **HYPOTHESIS:** (1) The daily EGARCH fit diverged. (2) A leak: chain implied vols were set from the true next-day variance while VIX carried noise.
* **CHANGES IMPLEMENTED:** (1) The daily EGARCH fit is now checked (finite, stationary, persistence in [0, 1), variance within a factor of e^5 of the sample), retried with normal errors, and falls back to EWMA. (2) The chain's ATM implied vol now equals the published noisy VIX.
* **EXPERIMENTS PERFORMED:** Traced one failing case (AR coefficient 14.7, asymmetry -874); reran the zero-effect cells.
* **RESULTS:** The retry fixes the failing case; the false GEX gain disappears.
* **IMPROVEMENTS:** The first 13 markets were discarded and the power study restarted; the discarded results are not reported anywhere.
* **REMAINING ISSUES:** The paper's EGARCH choice for the daily component should be compared with simpler daily models on real data.
* **NEXT ITERATION:** The before/after experiment.

## Iteration 9: the before/after experiment needed a better null

* **OBJECTIVE:** Test whether the paper's minute-level before/after tests are trustworthy (E2).
* **PROBLEM IDENTIFIED:** With persistent daily volatility, even day-level tests rejected "no change" 55 to 65% of the time when nothing about expiry had changed.
* **HYPOTHESIS:** Not a test bug: two separate three-month windows genuinely differ in volatility level, so the tests answer a different question from the paper's.
* **CHANGES IMPLEMENTED:** E2 has two scenarios: independent days (isolates within-day dependence) and persistent volatility with a week-relative difference-in-differences test.
* **EXPERIMENTS PERFORMED:** Both scenarios, many replications.
* **RESULTS:** The week-relative test holds its size; the minute-level test does not.
* **IMPROVEMENTS:** Design rule for the real study: compare each expiry day with the other days of its own week.
* **REMAINING ISSUES:** None for the simulation.
* **NEXT ITERATION:** Harden the daily-variance fallback.

## Iteration 10: a new test caught a zero-variance fallback

* **OBJECTIVE:** Test the EWMA fallback added in iteration 8.
* **PROBLEM IDENTIFIED:** On a degenerate series the fallback could return a daily variance of exactly zero.
* **HYPOTHESIS:** Every later normalisation would divide by zero.
* **CHANGES IMPLEMENTED:** EWMA starts from the training mean square and is floored at 1% of the overall mean square.
* **EXPERIMENTS PERFORMED:** New unit test; count of fallbacks in E1a.
* **RESULTS:** Test passes; the fallback was triggered in 0 of 180 fits.
* **IMPROVEMENTS:** Safe if it is ever needed.
* **REMAINING ISSUES:** None.
* **NEXT ITERATION:** Full synthetic pipeline run.

## Iteration 11: full pipeline run on one synthetic market

* **OBJECTIVE:** Check the whole pipeline end to end after the fixes.
* **PROBLEM IDENTIFIED:** None new; this was a confirmation run.
* **HYPOTHESIS:** The full run should agree with the power study.
* **CHANGES IMPLEMENTED:** None.
* **EXPERIMENTS PERFORMED:** `scripts/run_pipeline.py --source synthetic --days 500 --kappa 0.5 --seed 7` (183 s), reported as E7.
* **RESULTS:** A correctly signed GEX gave a small, borderline gain at 30 minutes; the US sign added nothing; the last hour could not separate models; the clearest gain came from the hybrid correction of mcsGARCH.
* **IMPROVEMENTS:** Confirmed that GEX must be judged against a strong baseline, not against mcsGARCH alone.
* **REMAINING ISSUES:** Everything so far is synthetic.
* **NEXT ITERATION:** Real data.

## Iteration 12: first contact with real NSE files (2 November 2026, evening)

* **OBJECTIVE:** Download real NSE data and get the real-data path running.
* **PROBLEM IDENTIFIED:** Several, each found by a check: (1) `www.nseindia.com` is blocked but `nsearchives.nseindia.com` works, and the old downloader needed the main site's cookies. (2) No free minute data from NSE. (3) A first OI check failed by a factor of 1.8. (4) Two lot sizes can be live on one day. (5) The VIX loader parsed ISO dates day-first. (6) Some April 2023 index files write the date month-first inside the file.
* **HYPOTHESIS:** (3) Expiring contracts stay in the bhavcopy on their last day but leave the participant file. (4) Converting OI with one lot size is wrong around lot changes.
* **CHANGES IMPLEMENTED:** Archive-only downloader (filters to NIFTY and BANKNIFTY index derivatives, also saves official index values and VIX). Minute bars from two MIT-licensed GitHub datasets plus `validate_minute_bars.py`. OI kept in shares with lot size 1 (GEX exact in rupees). Expiry days read from the bhavcopies. Dates taken from file names. Forward from put-call parity. Tests for each fix.
* **EXPERIMENTS PERFORMED:** Validation of minute bars against NSE daily values; OI unit check on 9 dates.
* **RESULTS:** 944 trading days (the 57 missing weekdays are all holidays). Nifty minute highs within 5 basis points of NSE on 100% of 943 days, lows on 99.9%; Nifty Bank 99.3 to 99.4% of 811 days. OI matches NSE's participant totals to within one contract on all 9 dates once expiring contracts are left out.
* **IMPROVEMENTS:** The prototype runs on real, checked data without a broker account.
* **REMAINING ISSUES:** Minute data have no volume before 2026; dealer signs are not observed.
* **NEXT ITERATION:** Full real-data forecasting run.

## Iteration 13: first full real-data forecasting contest

* **OBJECTIVE:** Run `scripts/run_pipeline.py --source nse` on 942 days (January 2023 to 2 November 2026) and compare models out of sample.
* **PROBLEM IDENTIFIED:** The question was open: does the paper's model hold up out of sample, and does GEX help?
* **HYPOTHESIS:** From the simulations, a hybrid (mcsGARCH corrected by LightGBM) should beat mcsGARCH, and GEX should add little.
* **CHANGES IMPLEMENTED:** `--start`, `--end`, `--min-coverage 0.95`, `--rate 0.065`, official trading days from NSE's index file; `daily_market.csv` written for day-level work. CI on GitHub Actions and a clean-install check.
* **EXPERIMENTS PERFORMED:** Walk-forward, expanding window, refit every 21 days, 792 test days; Diebold-Mariano tests and Model Confidence Set.
* **RESULTS:** (REAL DATA, after the fix in iteration 14.) Hybrid beats mcsGARCH by about 7% (next 15 minutes) and 9% (next 30 minutes), p < 0.001. HAR looked best for the last hour (about 11% better than mcsGARCH, p = 0.015), but see iteration 19. GEX adds nothing out of sample under any sign rule.
* **IMPROVEMENTS:** A real, honest headline that matches the simulation's warning.
* **REMAINING ISSUES:** In-sample relations between GEX and volatility untested; the paper's own tables not yet replicated.
* **NEXT ITERATION:** Day-level studies.

## Iteration 14: a 48.7% implied vol exposed an ordering bug

* **OBJECTIVE:** Build the demo notebook (`notebooks/01_real_data_tour.ipynb`).
* **PROBLEM IDENTIFIED:** The notebook showed an at-the-money implied vol of 48.7% on a calm day.
* **HYPOTHESIS:** UDiFF bhavcopies (from July 2024) are not sorted by strike, and missing implied vols were filled from neighbouring rows in file order, so they were filled from unrelated strikes.
* **CHANGES IMPLEMENTED:** `load_bhavcopy` sorts by expiry, option type and strike; test added. Pipeline and all day-level results regenerated.
* **EXPERIMENTS PERFORMED:** Compared old and new outputs.
* **RESULTS:** ATM implied vol on the same day 16.3%. Net GEX changed noticeably on about 100 days where it is near zero; the largest change in any scoreboard cell was 1.4 points; no conclusion changed.
* **IMPROVEMENTS:** A sanity number in a demo caught a bug that tests had not. (The pinning study later confirmed independently that 41 of 43 filled vols on one chain had been wrong before the fix.)
* **REMAINING ISSUES:** Other silent ordering assumptions could exist; the independent review looks for them.
* **NEXT ITERATION:** Day-level questions.

## Iteration 15: day-level expiry effects and GEX (R2 to R4)

* **OBJECTIVE:** Measure the expiry-day volatility premium under each expiry regime, and whether yesterday's GEX relates to today's volatility.
* **PROBLEM IDENTIFIED:** The participant-based GEX drifts upward over 2023 to 2026, so a GEX coefficient could just pick up the regime changes.
* **HYPOTHESIS:** Adding regime-specific levels and expiry effects removes that drift.
* **CHANGES IMPLEMENTED:** Week-relative expiry comparison by regime; regressions with and without regime x expiry controls; a crude pinning check.
* **EXPERIMENTS PERFORMED:** `experiments/real_day_level.py`.
* **RESULTS:** Expiry days carry about 11% more variance before November 2024, no clear difference in the Thursday single-weekly regime, and about 21% more since the move to Tuesday (43% in the last hour). In-sample, higher GEX goes with calmer next days on ordinary days but with more volatile expiry days, under all three sign rules and with regime controls. The crude pinning check found nothing.
* **IMPROVEMENTS:** A new, testable pattern (the expiry-day sign flip).
* **REMAINING ISSUES:** In-sample only; sign of dealer positions is a guess; several specifications tried.
* **NEXT ITERATION:** Test a mechanism from the literature (hedging-driven end-of-day momentum).

## Iteration 16: end-of-day momentum and dealer gamma (R5)

* **OBJECTIVE:** Test whether Nifty shows the US "intraday momentum" pattern and whether it is stronger when dealers are short gamma (Baltussen et al. 2021).
* **PROBLEM IDENTIFIED:** The first run found the opposite: a reversal, concentrated on negative-GEX days under the Pro rule.
* **HYPOTHESIS:** Either the effect is real, or it is driven by volatile periods (reversal is stronger in busy markets), or by the GEX drift across regimes.
* **CHANGES IMPLEMENTED:** Added regime-specific slopes, VIX and move-size controls, and a by-regime and by-half stability table to `experiments/momentum_gex.py`.
* **EXPERIMENTS PERFORMED:** 932 days, Newey-West errors, three sign rules, all, non-expiry and expiry days.
* **RESULTS:** Overall slope -0.029 (p = 0.015). Pro rule interaction -0.068 (p = 0.0002), -0.050 (p = 0.003) with VIX and size controls, same sign in all three regimes and both halves. The client-mirror result (38 days) disappears with the controls.
* **IMPROVEMENTS:** One result discarded as a volatility artefact; one survives several checks but contradicts the hedging story under the same sign that supports R3.
* **REMAINING ISSUES:** The R3 and R5 signs cannot both fit one simple hedging story; the falsification suite is designed to separate the explanations.
* **NEXT ITERATION:** Replicate the paper itself.

## Iteration 17: replicating the paper's Tables 1 to 6 on NSE data

* **OBJECTIVE:** Rebuild every table of the paper from real data.
* **PROBLEM IDENTIFIED:** The stated "before" window (November 2023 to February 2024) has 82 trading days, not the paper's 64, and its statistics are far from Tables 1 and 2. Also two bugs in `volatility.py`: an all-missing minute column made mcsGARCH silently return its starting values, and `diurnal()` with no argument crashed.
* **HYPOTHESIS:** The "before" sample may be a different 64-day window.
* **CHANGES IMPLEMENTED:** `experiments/replicate_paper.py`, `src/niftygex/replication.py`, `docs/REPLICATION_NSE.md`; a search over every 64-day window from July 2022 to February 2024; both `volatility.py` bugs fixed with regression tests.
* **EXPERIMENTS PERFORMED:** Tables 1 to 6 on the stated and inferred windows, a 40-setting grid for connectedness, day-level and cluster-bootstrap tests next to the paper's minute-level ones.
* **RESULTS:** The "after" sample replicates closely (exactly the paper's 21,239 Nifty returns; mcsGARCH alpha 0.0498 and beta 0.9407 against printed 0.0494 and 0.9410). The window 14 November 2022 to 10 February 2023 reproduces the printed "before" maxima to every printed decimal and agrees on statistics not used in the search. The fall in persistence and in overall connectedness replicate; hour-level connectedness and the Bank Nifty Wednesday volatility result do not survive robust tests.
* **IMPROVEMENTS:** A concrete, checkable question for the authors, and robust versions of every test.
* **REMAINING ISSUES:** The inferred window is an inference, not a confirmation.
* **NEXT ITERATION:** A proper pinning study.

## Iteration 18: pinning study, and a holiday-expiry bug

* **OBJECTIVE:** Replace the crude pinning check with a pre-specified study.
* **PROBLEM IDENTIFIED:** Contracts listed for Thursday 29 June 2023 really expired on Wednesday 28 June (the holiday moved), so their time to expiry was overstated on 26 and 27 June 2023.
* **HYPOTHESIS:** Pinning, if present, should be stronger when dealers are long gamma near the strike.
* **CHANGES IMPLEMENTED:** `experiments/pinning_study.py` and `docs/PINNING_STUDY.md`; `data.holiday_adjusted_expiries` maps a listed expiry that is a holiday to the previous trading day, with a test.
* **EXPERIMENTS PERFORMED:** Distance from the official close to the nearest strike on expiry days versus other days, by regime, for Nifty and Bank Nifty; permutation tests; a dealer-gamma split with a power calculation.
* **RESULTS:** Pooled Nifty: 1.10 points closer on expiry days, p = 0.054. Thursday single-weekly regime: 4.52 points closer, p = 0.0006; none since the move to Tuesday. Bank Nifty with weekly options: 4.26 points closer, p = 0.008. No link to any dealer-gamma proxy (Holm p = 0.52 to 0.91). Every expiring option settles at the official close on all 322 expiries checked.
* **IMPROVEMENTS:** A careful null on gamma and a regime-dependent pinning result.
* **REMAINING ISSUES:** The half-year pattern was seen after looking at the data, so it is exploratory.
* **NEXT ITERATION:** Independent review and falsification (in progress at the time of writing).

## Iteration 19: NSE's closing auction changes the last minutes of the day

* **OBJECTIVE:** Prepare a live forecast for the 3 November 2026 expiry from the latest data.
* **PROBLEM IDENTIFIED:** The 2 November minute bars showed the index frozen from 15:15 to 15:28 and then jumping 47 basis points in the 15:29 bar. The pinning study had separately found that NSE moved to a closing auction on 3 August 2026.
* **HYPOTHESIS:** For the last 63 days, "last hour" and "last 30 minutes" contain dead minutes and an auction jump, which could drive results that use the end of the day (R1 last hour, R2 last hour, R5).
* **CHANGES IMPLEMENTED:** `experiments/closing_auction_check.py` documents the auction in the data and reruns R1, R2 and R5 without those days.
* **EXPERIMENTS PERFORMED:** Frozen-minute share, jump size and match with the official close; scoreboards and Diebold-Mariano tests on test days before 3 August 2026; Tuesday regime split; R5 regressions without the auction days.
* **RESULTS:** The auction is clear in the data (frozen on 96% of late minutes, 15:29 close equal to NSE's official close on all 63 days). The hybrid's 15- and 30-minute gains, the Tuesday expiry effect and the R5 interaction are unchanged without those days. **HAR's last-hour edge is not:** 10.8% (p = 0.015) on all test days, but 4.2% (p = 0.11) before the auction. The headline was corrected.
* **IMPROVEMENTS:** One overstated claim removed before anyone relied on it.
* **REMAINING ISSUES:** Forecast models for the end of the day should treat the auction period as a separate regime; the model study is told about it.
* **NEXT ITERATION:** Live dry run for the 3 November expiry.

## Iteration 20: an independent review of the real-data results

* **OBJECTIVE:** Have a reviewer who did not write the code check every real-data number against the result files and look for look-ahead bias.
* **PROBLEM IDENTIFIED:** Seven main points (M1 to M7, `docs/INDEPENDENT_REVIEW.md`). The biggest: (M1) under squared-error loss the R1 ranking reverses; (M2) the hybrid's gain might all come from the 09:15 anchor; (M3) the refit schedule was described wrongly; (M5) the R3 GEX ratio is about four times larger on expiry eves because gross gamma peaks then, which could create the expiry-day "flip" mechanically; (M6) many tests, no correction.
* **HYPOTHESIS:** Some headline claims are artefacts of loss choice, anchor time or scaling.
* **CHANGES IMPLEMENTED:** `experiments/forecast_diagnostics.py` (MSE, gains by anchor and regime, trimmed losses); `r3_robustness` in `experiments/real_day_level.py` (scale-free GEX, gross-gamma controls, Holm and Benjamini-Hochberg); Holm and BH columns in R5; corrected wording throughout `docs/REAL_DATA_RESULTS.md`.
* **EXPERIMENTS PERFORMED:** All of the above on the committed forecast files and panels.
* **RESULTS:** (M1) The MSE reversal comes from about 10 anchors right after the 4 June 2024 election-result crash, which make up 97% of mcsGARCH's squared error. (M2) The hybrid's gain is largest at 09:15 (16.4% and 26.9%) but still 5.8% and 7.1% at later anchors and in every regime. (M5) With scale-free GEX the R3 flip survives only under the Pro rule, so "holds under all three rules" was withdrawn.
* **IMPROVEMENTS:** Two overstated claims corrected and every remaining claim now says which loss, anchor and sign rule it depends on.
* **REMAINING ISSUES:** The Diebold-Mariano test should use a Student t reference and the hybrid's error model uses a full-sample constant (handed to the model study).
* **NEXT ITERATION:** Try to break the two GEX findings that are left.

## Iteration 21: the expiry rule changes as natural experiments

* **OBJECTIVE:** Use NSE's changes to the expiry calendar (November 2024, September 2025) as events, with Bank Nifty as a comparison index.
* **PROBLEM IDENTIFIED:** R2's regime comparison is a before/after comparison with no control group.
* **HYPOTHESIS:** If expiry-day hedging adds volatility, the expiry premium should move with the rule changes in Nifty and not in the comparison index.
* **CHANGES IMPLEMENTED:** `experiments/expiry_event_study.py`, `docs/EVENT_STUDY.md`.
* **EXPERIMENTS PERFORMED:** Week-relative difference-in-differences, placebo event dates, a pre-specified main test, Rogers-Satchell range-based variance next to 5-minute realized variance.
* **RESULTS:** The pre-specified main test (Bank Nifty weekly options stopping in November 2024, Nifty as the comparison) is null: -0.01 log points, p = 0.97, though it could only have detected large effects. The end-of-day premium does follow the expiry: when Bank Nifty's expiry moved from Thursday to Wednesday, the Wednesday-minus-Thursday gap in its last 30 minutes rose by +1.14 log points (placebo p = 0.005) while whole-day variance did not move. Rogers-Satchell variance shows an expiry premium that 5-minute realized variance does not.
* **IMPROVEMENTS:** An honest null on the main question and a narrower, better-supported end-of-day result.
* **REMAINING ISSUES:** BSE moved Sensex expiries at the same time and BSE data are blocked from our servers, so a Sensex confound cannot be ruled out.
* **NEXT ITERATION:** Falsification of F1 and F2.

## Iteration 22: trying hard to break the GEX findings

* **OBJECTIVE:** Attack F1 (R3: GEX and next-day volatility, with an expiry-day flip) and F2 (R5: stronger end-of-day reversal after negative Pro GEX) with every test we could think of.
* **PROBLEM IDENTIFIED:** Both findings came from in-sample regressions with a guessed dealer sign, three rules and several specifications.
* **HYPOTHESIS:** If a finding is a dealer-hedging effect, it should survive shuffled GEX, fail when GEX comes from the future, not be reproduced by random sign rules or gross gamma, hold out of sample and replicate on Bank Nifty.
* **CHANGES IMPLEMENTED:** `experiments/falsification.py` (seven test families, 742 tests in total, all seeds recorded), `tests/test_falsification.py`, `docs/FALSIFICATION.md`. While reviewing it, two small bugs in R5 were found and fixed: yesterday's VIX was lagged after dropping days (stale on 4 days) and days with no GEX were coded as "non-negative" (3 days). R5 changed only in the fourth decimal.
* **EXPERIMENTS PERFORMED:** Five permutation schemes, timing placebos (GEX from t, t+1, t+2), a cloud of random sign rules, gross GEX and open interest as placebos, a 2023-24 / 2025-26 split with a pre-registered test, Bank Nifty, a quasi-causal test using the disappearance of expiring gamma, and Holm corrections.
* **RESULTS:** **F2 survived every test** (permutations p = 0.0005 to 0.010; GEX from the next day does not predict it, p = 0.31; out-of-sample squared error 4.3% lower, p = 0.007; Holm p = 0.016 among 78 tests). **F1's ordinary-day term failed** (block shuffle p = 0.25; 91.9% of random sign rules also significant; gross gamma alone gives t = -5.3), so it is a gross-gamma level effect. **F1's expiry-day flip** survived in sample under the Pro rule only, makes forecasts about 5% worse and does not replicate on Bank Nifty. The quasi-causal roll-off test found nothing distinguishable from placebo days. 16 of 742 tests survive Holm.
* **IMPROVEMENTS:** One finding promoted to "robust", one demoted to "proxy for gross gamma", one to "in-sample lead".
* **REMAINING ISSUES:** The Pro GEX series is driven by the daily Pro positioning weights, not by the option chain, so what it measures is still unclear.
* **NEXT ITERATION:** Estimate the dealer sign from the data instead of guessing it.

## Iteration 23: estimating the dealer sign from the market's response

* **OBJECTIVE:** Let the data choose the dealer sign: write dealer gamma as a weighted mix of the participant categories (and of calls and puts) and estimate the weights from two outcomes that hedging theory makes predictions about.
* **PROBLEM IDENTIFIED:** R3 fits the Pro sign and R5 fits the reversed Pro sign. Both cannot be right under one simple story.
* **HYPOTHESIS:** If there is one dealer sign, the variance outcome (a) and the end-of-day outcome (b) should point to the same weights.
* **CHANGES IMPLEMENTED:** `experiments/dealer_sign.py` (direction estimates with likelihood-ratio and bootstrap sets, a slowly drifting state-space version, expanding-window out-of-sample test, shuffled-outcome placebos and a synthetic power check), `tests/test_dealer_sign.py` (18 tests), `docs/DEALER_SIGN.md`. GEX is scale-free throughout.
* **EXPERIMENTS PERFORMED:** Full-sample and rolling estimates, 452 out-of-sample days refitted every 21 days with Diebold-Mariano tests, 200 placebos, synthetic data with known weights.
* **RESULTS:** A clear negative result. Outcome (a) points to 21 degrees (mostly Pro) but weakly (p = 0.18); outcome (b) points to 190 degrees, the reverse of Pro, strongly (p = 4e-6, placebo p = 0.005). Out of sample, all 10 primary tests have Holm p = 1.0. The method finds known weights on synthetic data (95% detection, 9 degree median error at signal-to-noise 0.10); the real outcome (a) is below that range. The old R3 scaling produces a false call/put direction (52 degrees, p = 0.013) that disappears with the scale-free measure.
* **IMPROVEMENTS:** The sign question is now stated as a measured, documented result rather than an assumption.
* **REMAINING ISSUES:** Public NSE data do not identify who hedges Nifty options. A dealer-level or strike-level positioning dataset would be needed.
* **NEXT ITERATION:** Scoring the live forecast for 3 November and a one-command reproduction.

## Iteration 24: a frozen live forecast and a one-command reproduction

* **OBJECTIVE:** Show the method works on a day nobody has seen yet, and make every real-data result reproducible with one command.
* **PROBLEM IDENTIFIED:** Every result so far is on past data, and the raw data are not in the repository (licence and size).
* **HYPOTHESIS:** A forecast committed before the market opens cannot be tuned after the fact; hashes of the raw files let anyone confirm they have the same data.
* **CHANGES IMPLEMENTED:** `experiments/live_dry_run.py` (freeze and score modes); the forecast for Tuesday 3 November 2026 was committed at 21:01 UTC on 2 November, before the market opened (`results/live/2026-11-03/forecast.md`). `scripts/data_manifest.py` and `data/MANIFEST.csv` (SHA-256 of 2,989 raw files; gzip files hashed on their content because gzip headers carry a time stamp). `scripts/reproduce_all.py` downloads with pinned sources, checks the manifest and reruns every study, logging run times.
* **EXPERIMENTS PERFORMED:** Manifest check on the current data (passes); partial reproduction run.
* **RESULTS:** Frozen forecast: whole-day variance 5.55e-05 (about 11.8% annualised volatility), last hour 1.33e-05, ATM straddle 152 points at strike 23,350, all three GEX rules positive. Scoring after the close is scheduled.
* **IMPROVEMENTS:** An out-of-sample test that is out of sample in calendar time, not just in the code.
* **REMAINING ISSUES:** One day is an illustration, not evidence; the full reproduction run waits for the model study to finish.
* **NEXT ITERATION:** Merge the model study, rerun the pipeline with the holiday and monthly-expiry fixes, and run the full reproduction.

## Iteration 25: gamma at the current index level, minute by minute

* **OBJECTIVE:** Test the hedging story where it should be strongest: on expiry day, when the gamma of options near the current index level changes minute by minute as the index moves and the clock runs down.
* **PROBLEM IDENTIFIED:** Every earlier GEX measure was one number per day, computed at the previous close. Hedging pressure on expiry day depends on where the index is now.
* **HYPOTHESIS:** If dealers are net long (short) gamma near spot, the next minutes should be calmer (more volatile) when gamma at spot is high, after removing each day's level and the normal intraday pattern.
* **CHANGES IMPLEMENTED:** `experiments/intraday_gamma.py`, `tests/test_intraday_gamma.py`, `docs/INTRADAY_GAMMA.md`. Gamma at spot is rebuilt at every anchor from the previous evening's open interest and the current index level and time to expiry, with day and minute-of-day fixed effects and three placebos (strikes shifted by half a spacing, another day's open-interest profile, permuted open interest). Two follow-ups were pre-registered in `results/intraday_gamma/design_log.csv` before their code was written.
* **EXPERIMENTS PERFORMED:** 15-minute anchors from 09:30 to 15:00 on all day types; an out-of-sample last-hour test; then the final half hour of expiry days (one anchor per minute, 15:00 to 15:24, before the closing auction) and a Bank Nifty replication of the one lead.
* **RESULTS:** A clean null on expiry days (0.142, interval -0.128 to 0.413). The significant coefficients on other days are the leverage effect (falling markets are more volatile), not hedging: placebos give the same numbers. Gamma concentration near spot passed all three placebos on Nifty (-0.386, p = 0.027) but failed both pre-registered follow-ups: the final half hour (0.257, p = 0.39) and Bank Nifty (-0.122, p = 0.66). Holm over the follow-ups gives 1.0. Out of sample, gamma at 14:30 changes last-hour QLIKE by -1.8% (p = 0.53).
* **IMPROVEMENTS:** The most direct test of the mechanism that public data allow, with its limits measured: before 15:00 the reach of each option's gamma is wider than one strike spacing, so gamma at spot is a smooth profile and cannot separate real strikes from placebos.
* **REMAINING ISSUES:** Positions during the day are not public. Intraday option-chain snapshots would be needed for a sharper test.
* **NEXT ITERATION:** Merge the model study and rerun the pipeline.

## Iteration 26: a blind second transcription of every number taken from the paper (by a separate AI agent)

* **OBJECTIVE:** Make sure the replication compares our numbers with what the paper actually printed.
* **PROBLEM IDENTIFIED:** The 223 printed values in `experiments/replicate_paper.py` were typed by hand once. A typing slip could make the replication look worse, or make the paper look inconsistent when it is not, just before we raise inconsistencies with the authors.
* **HYPOTHESIS:** A blind second typing, compared cell by cell, finds any slip.
* **CHANGES IMPLEMENTED:** `results/replication_nse/rekey_blind.csv` (typed from the PDF before the script's values were opened), `experiments/rekey_check.py`, `results/replication_nse/rekey_check.csv`, `docs/REKEY_CHECK.md`. The paper itself stays out of the repository.
* **EXPERIMENTS PERFORMED:** Cell-by-cell comparison of all 223 values; every typed table value also found in the PDF's own text layer (217 of 217).
* **RESULTS:** 222 of 223 matched. The one difference is a typesetting slip in the paper, which the first typing had already read correctly. Nothing in the replication changed.
* **IMPROVEMENTS:** The questions for the authors now rest on a checked transcription.
* **REMAINING ISSUES:** None for the transcription.
* **NEXT ITERATION:** Connectedness by gamma regime.

## Iteration 27: does volatility spill over more when dealers are short gamma?

* **OBJECTIVE:** Link the paper's connectedness result (Nifty and Bank Nifty became less connected after the expiry split) to dealer hedging.
* **PROBLEM IDENTIFIED:** The paper measures connectedness but does not ask what drives it day to day.
* **HYPOTHESIS:** After short-gamma evenings (negative GEX), hedging amplifies moves in both indices, so total connectedness is higher.
* **CHANGES IMPLEMENTED:** `experiments/connectedness_gamma.py`, `docs/CONNECTEDNESS_GAMMA.md`, results in `results/connectedness_gamma/`. The test was stated before it was run, but that statement was not time-stamped in git; the document says so.
* **EXPERIMENTS PERFORMED:** Diebold-Yilmaz connectedness on 1-minute Nifty and Bank Nifty volatility, split by the evening's GEX sign under each rule, with permutation p-values, a day-level regression with VIX and robustness checks.
* **RESULTS:** The pre-stated Pro-rule test is borderline: +4.29 points of connectedness after short-gamma evenings, permutation p = 0.044. It vanishes once VIX is controlled for, and no Pro secondary test survives Holm. Gross gamma (no sign) does survive (Holm p = 0.010), with lower connectedness on high-gamma days: the same level effect as in iteration 22.
* **IMPROVEMENTS:** A new question for the paper's own design, answered with its limits stated.
* **REMAINING ISSUES:** The Nifty and Sensex pair is still open.
* **NEXT ITERATION:** BSE's Sensex expiry as a confounder.

## Iteration 28: BSE's Sensex expiry as a hidden confounder

* **OBJECTIVE:** Check whether Sensex expiry days, which BSE moved around during the sample, contaminate the "rest of the week" baseline in R2.
* **PROBLEM IDENTIFIED:** If Sensex expiry days are volatile for Nifty too, they inflate the baseline and bias the Nifty expiry premium.
* **HYPOTHESIS:** Nifty volatility is higher on Sensex expiry days; leaving them out changes R2.
* **CHANGES IMPLEMENTED:** `experiments/sensex_confound.py` (with `--sensex-ref` pinning the Sensex minute-bar commit, `--until` and `--fetch-only`), `data/external/sensex_expiry_calendar.csv`, `docs/SENSEX_CONFOUND.md`. BSE's servers are blocked from the cloud, so the calendar is built from reported dates and checked against the 71 option-price expiry dates available (all match).
* **EXPERIMENTS PERFORMED:** The pre-stated test of Nifty volatility on Sensex expiry days; R2 rerun with Sensex days out of the baseline.
* **RESULTS:** The pre-stated test is not significant (p = 0.16), but Nifty's last 30 minutes are more volatile on Sensex expiry days. With Sensex days out of the baseline every NSE premium gets a little larger (Tuesday last hour +43% becomes +63%), so R2 was understated, not overstated.
* **IMPROVEMENTS:** One more alternative explanation checked and reported.
* **REMAINING ISSUES:** The calendar needs verification against BSE circulars, downloaded off the cloud.
* **NEXT ITERATION:** Power calibrated on the real data.

## Iteration 29: how large an effect could our tests see?

* **OBJECTIVE:** Turn "not significant" into "we could only have seen effects larger than X", and check that the main tests hold their 5% level.
* **PROBLEM IDENTIFIED:** Several central results are nulls, and the significant ones come from a sample of about 200 expiries.
* **HYPOTHESIS:** Resampling the real residuals shows the minimum detectable effects and the size of each test.
* **CHANGES IMPLEMENTED:** `experiments/power_real.py`, `docs/POWER_REAL.md`, results in `results/power_real/`.
* **EXPERIMENTS PERFORMED:** Block resampling of real residuals with effects of known size planted, for R2, R3, R5 and the out-of-sample GEX test; sign-error scenarios; size checks with no effect.
* **RESULTS:** The significant results sit just above what the sample can detect (R5: detectable 0.042 to 0.052 against an estimate of -0.068). Effects half as large would need 2.4 to 3.9 times the sample (5.5 to 10.9 more years). A GEX sign wrong on a quarter of days cuts power at the estimate from 83% to 99% down to 23% to 59%. R3, R5 and the Tuesday R2 tests hold their level, but the R2 tests for the earlier periods over-reject (several weeklies 8.2% to 9.1%, one Thursday weekly 12.8% to 17.8% at a nominal 5%), so the "+11% before November 2024" result (p = 0.045) is weak.
* **IMPROVEMENTS:** Every null now has a stated detectable size; one result was downgraded.
* **REMAINING ISSUES:** Power is for the designs as they are; a larger sample needs time, not code.
* **NEXT ITERATION:** The two pre-registered studies.

## Iteration 30: what explains when pinning appears (pre-registered, P87)

* **OBJECTIVE:** Explain why pinning appeared under the single Thursday weekly and not since the move to Tuesday.
* **PROBLEM IDENTIFIED:** The pinning study (iteration 18) found the pattern but no cause, and dealer gamma did not explain it.
* **HYPOTHESIS:** Six market-structure candidates written down in advance; the primary one is open interest near the money.
* **CHANGES IMPLEMENTED:** The analysis plan was committed to git (`docs/prereg/P87_PINNING_MECHANISM.md`, commit e50a775) before this study's outcomes were computed, on the same data that had shown the pinning pattern; a second agent then ran it. `experiments/pinning_mechanism.py`, `docs/PINNING_MECHANISM.md`, `data/external/other_weekly_expiries.csv`, results in `results/pinning_mechanism/`.
* **EXPERIMENTS PERFORMED:** The pre-registered regressions with week-clustered and permutation p-values, Holm across candidates, the planned placebos and a settlement-price check (322 of 322 match).
* **RESULTS:** The primary candidate has the wrong sign (+0.0224 strike spacings per SD, p = 0.010), so it is ruled out. Open interest held into expiry rather than rolled matters day to day (Holm p = 0.0099) but explains none of the change between regimes. No candidate explains when pinning comes and goes. No deviations from the plan.
* **IMPROVEMENTS:** The first result in the project whose order (design, then data) anyone can check from git.
* **REMAINING ISSUES:** Pinning's cause is still unknown; strike-level positioning data would help.
* **NEXT ITERATION:** Is the end-of-day reversal worth anything after costs?

## Iteration 31: the end-of-day reversal after trading costs (pre-registered, P90)

* **OBJECTIVE:** Measure whether the one robust GEX result (R5) is economically meaningful.
* **PROBLEM IDENTIFIED:** A statistically robust effect can still be too small to matter.
* **HYPOTHESIS:** A walk-forward trading rule fixed in advance earns more than realistic costs.
* **CHANGES IMPLEMENTED:** The analysis plan was committed to git (`docs/prereg/P90_F2_ECONOMICS.md`, commit 236e3db) before this study's returns were loaded; the test period had already been used to find R5, as the plan itself says. `experiments/f2_economics.py`, `docs/F2_ECONOMICS.md`, results in `results/f2_economics/`.
* **EXPERIMENTS PERFORMED:** The pre-registered rule on 90 test trades from 2025 to July 2026, with gross and net returns, cost scenarios, the Deflated Sharpe Ratio and the planned robustness checks.
* **RESULTS:** +4.15 bp per trade before costs, -0.89 bp after a base-case 5 bp round trip; the Deflated Sharpe Ratio is far below the 0.95 bar. By the pre-stated rule, not economically meaningful.
* **IMPROVEMENTS:** R5 is now described as a statistical regularity, not a trading opportunity, everywhere it appears.
* **REMAINING ISSUES:** The option side of a trading layer still needs intraday option prices.
* **NEXT ITERATION:** Count every test in the project.

## Iteration 32: a ledger of every hypothesis test

* **OBJECTIVE:** Show how much of the evidence survives once every test the project ran is counted.
* **PROBLEM IDENTIFIED:** Each study corrected for its own tests, but the project as a whole ran thousands.
* **HYPOTHESIS:** A Benjamini-Yekutieli correction across the whole project (valid under any dependence) separates the robust findings from the lucky ones.
* **CHANGES IMPLEMENTED:** `scripts/build_test_ledger.py`, `results/test_ledger*.csv`, `docs/TEST_LEDGER.md`, with a pre-specification flag judged from git commit times.
* **EXPERIMENTS PERFORMED:** 4,930 tests from 17 studies; BY across the project, BY and Holm within each study; a reconciliation of the eight headline findings.
* **RESULTS:** 747 tests survive project-wide BY (540 survive Holm within their study). The forecasting result survives in full; the end-of-day reversal survives in part; the expiry-day sign flip does not. The ledger also found that "pre-stated" tests in two studies were not time-stamped in git, which the documents now say.
* **IMPROVEMENTS:** A reader can check any claim against its corrected p-value in one place.
* **REMAINING ISSUES:** The ledger is being rebuilt to count the second forecasting contest (iteration 33) test by test.
* **NEXT ITERATION:** Merge the stronger forecasting study.

## Iteration 33: a stronger forecasting contest (the model study)

* **OBJECTIVE:** Give the paper's model and GEX a fair chance: refit mcsGARCH as often as the challengers, add option-implied information, and test GEX against a strong base.
* **PROBLEM IDENTIFIED:** In the first contest mcsGARCH was refitted every 63 days and the challengers every 21; the base features had no option-implied information, so GEX might have looked useless (or useful) for the wrong reason; the last-hour models had one training row per day; two small look-ahead problems (a whole-sample scale, a placebo drawn from later days).
* **HYPOTHESIS:** With a fairer benchmark and a stronger base, the hybrid's edge shrinks but survives, and GEX still adds nothing.
* **CHANGES IMPLEMENTED:** `src/niftygex/implied.py` and changes to the feature, forecast, evaluation and pipeline code (tests `tests/test_forecast_v2.py`, `tests/test_implied.py`, including one that deletes the last days of a market and checks no earlier feature changes). Design chosen on test days up to 30 June 2025 only; the hold-out run once with the final configuration (the first run had covered those days). Output `results/pipeline_nse_v2/`, write-up `docs/MODEL_ITERATIONS_REAL.md`. Every study now reads its daily panels from the v2 run; the first run is kept in `results/pipeline_nse/` as an archive.
* **EXPERIMENTS PERFORMED:** 32 models, three horizons, 28 subsets (expiry days, before the auction, hold-out, regimes, the 09:15 anchor), QLIKE and MSE, a clean GEX ablation with a placebo, 5-seed averaging and a seed-noise study.
* **RESULTS:** (REAL DATA.) The 21-day refit makes mcsGARCH 1.6% to 2.1% better. Against it, the hybrid is 5.8%, 7.6% and 6.4% better at 15 minutes, 30 minutes and the last hour (all p < 0.001), 5.5%, 6.9% and 7.0% before the auction, and 6.4%, 9.2% and 7.1% on the hold-out. The last-hour result replaces "nothing reliably beats mcsGARCH"; it is the least secure horizon (54% of days, nothing in the Thursday regime). GEX changes the hybrid's loss by -0.4% to +0.1% at 15 and 30 minutes and makes the last hour worse; nothing survives Holm. A single LightGBM seed moves a model by up to 0.4 points, so sub-1-point differences are noise.
* **IMPROVEMENTS:** The headline forecasting claim now holds against a stronger benchmark and on a hold-out the design choices were not tuned on (the first run had covered those days, so it is not unseen data). Which protocol change produced the last-hour edge has not been isolated.
* **REMAINING ISSUES:** The live forecast of 3 November was frozen from the first run, so it is scored with that model.
* **NEXT ITERATION:** Regenerate every result from the raw downloads with one command and compare with the committed outputs.

## Iteration 34: the Nifty and Sensex expiry-weekday swap as a natural experiment (pre-registered, P92)

* **OBJECTIVE:** Use the one rule change that holds the weekday fixed: on 1 September 2025 Nifty moved its weekly expiry to Tuesday and Sensex to Thursday, the reverse of before.
* **PROBLEM IDENTIFIED:** Every earlier expiry comparison mixes the expiry with weekday effects. Comparing the same weekday before and after the swap removes them.
* **HYPOTHESIS:** On its own expiry day the expiring index becomes the net transmitter of minute-level volatility spillovers (Diebold-Yilmaz, the anchor paper's connectedness measure on a new pair).
* **CHANGES IMPLEMENTED:** The design was committed to git (`docs/prereg/P92_SENSEX_SWAP.md`, commit 042d510) before any connectedness or volatility of this study was computed. `experiments/sensex_swap.py`, `tests/test_sensex_swap.py`, `docs/SENSEX_SWAP.md`, results in `results/sensex_swap/`.
* **EXPERIMENTS PERFORMED:** A difference in differences of NET spillover (Tuesday minus Thursday, after minus before) with a block permutation over weeks as the decisive p-value; secondaries with Holm; placebos (Monday minus Wednesday, two fake swap dates); robustness checks.
* **RESULTS:** (REAL DATA.) Whole day: +0.42 points, permutation p = 0.33, not supported; effects as large as the pre-registered detectable size (4.90) are ruled out. Last hour: +2.51, Holm p = 0.015, in the predicted direction. But the design check failed: a fake swap in May 2025 moved the measure by +4.45 (p = 0.038) and Monday minus Wednesday moved by -4.96 (p = 0.005), so weekday gaps in this measure drift by about the size of the effect sought. No deviations from the plan.
* **IMPROVEMENTS:** The anchor paper's connectedness question asked on a new pair with the cleanest available design, and its weakness found by its own pre-stated placebos.
* **REMAINING ISSUES:** The last-hour sign needs new data; the Sensex calendar and bars are not verified against BSE.
* **NEXT ITERATION:** Regenerate everything from the raw files and rebuild the test ledger.

## Iteration 35: answering an AI referee on the end-of-day reversal (F2)

* **OBJECTIVE:** Take the hardest questions a separate AI agent asked, acting as a journal referee, about the one GEX result that survived falsification (F2).
* **PROBLEM IDENTIFIED:** The referee pointed out that the negative-Pro state lasts for weeks at a time, so ordinary standard errors overstate the evidence; that the cleanest test of hedging theory (dealers short all options) had not been put first; that only one participant category had been tried as the dealer proxy; and that the claim "the two dealer-sign outcomes are 169 degrees apart" was never tested.
* **HYPOTHESIS:** If F2 is a dealer-hedging effect, it should appear without a proxy, under other proxies, and under inference that respects the slow state.
* **CHANGES IMPLEMENTED:** `experiments/referee_f2.py`, `tests/test_referee_f2.py`, `docs/REFEREE_RESPONSE_F2.md`, results in `results/referee_f2/`, SEBI dates with sources in `data/external/sebi_phases.csv`. All post hoc.
* **EXPERIMENTS PERFORMED:** Seven items: the unconditional end-of-day slope with five kinds of standard error; the runs and effective sample size of the Pro sign; F2 with episode and quarter clusters, wild bootstraps and block permutations; FII, DII and Client rules; controls for the 15:15 split, stale prints, seasonality, SEBI phases and Sensex days; a test of the two dealer-sign angles; minute-level autocorrelation near the close.
* **RESULTS:** (REAL DATA, 598 unadjusted p-values.) The proxy-free test fails: the end-of-day slope is -0.029 (block-bootstrap interval -0.048 to -0.005), a reversal rather than the momentum hedging predicts, and it does not grow with gamma. F2 is robust in the full sample (largest robust p 0.012) and from 2025 (0.038), but not inside 2023 to 2024 (0.095 to 0.18). Only the Pro rule shows it. Controls leave it at -0.053 to -0.079. The two dealer-sign outcomes are not shown to differ (p 0.07 to 0.28 once the weak outcome may have no effect), so that claim was withdrawn in the pitch, README, briefing and `docs/DEALER_SIGN.md`.
* **IMPROVEMENTS:** F2 is now described as what it is: a slow Pro-positioning regime with a few dozen independent episodes.
* **REMAINING ISSUES:** Pro's raw call, put and futures positions are not yet tested separately; SEBI and Sensex dates are not verified against primary documents (hosts blocked).
* **NEXT ITERATION:** The forecasting and expiry-identification parts of the referee report.

## Iteration 36: did ending weekly options calm the old expiry weekday? (registered plan, P96)

* **OBJECTIVE:** Use SEBI's November 2024 cut (Bank Nifty, Fin Nifty and Midcap Select lost their weekly options) as a natural experiment for the anchor paper's question: does an expiry make its weekday more volatile?
* **PROBLEM IDENTIFIED:** A before-and-after comparison of one index mixes the rule change with everything else that changed in markets at the same time.
* **HYPOTHESIS:** Against a synthetic control built from option-free sector indices, Bank Nifty's Wednesday (its old expiry day) becomes calmer after the cut.
* **CHANGES IMPLEMENTED:** Plan committed before any outcome was computed (`docs/prereg/P96_SYNTHETIC_CONTROL.md`, commit 9634054), with the downloaded NSE files locked by a SHA-256 fingerprint. `experiments/synthetic_control.py`, `tests/test_synthetic_control.py` (23 tests on simulated data), `docs/SYNTHETIC_CONTROL.md`, results in `results/synthetic_control/`.
* **EXPERIMENTS PERFORMED:** Synthetic control (Abadie, Diamond and Hainmueller 2010) on weekly week-relative Parkinson variance; placebo ranks over 19 donors; a fake date in time; 8 registered sensitivity variants for each of the three indices.
* **RESULTS:** (REAL DATA.) No support: Bank Nifty's gap moved +0.12 log points, the wrong sign (placebo p = 0.95); Fin Nifty +0.28 and Midcap Select +0.37 (Holm 1.00). No sensitivity p below 0.20. The design could only detect effects of about 1.44 log points, so this is "too imprecise to tell", as the plan warned. Post hoc: the option-free sector indices had a Wednesday premium of their own before the cut (+0.28, Bank Nifty +0.19) that fell market-wide afterwards. No deviations from the plan.
* **IMPROVEMENTS:** Whole-day weekday premia are shown not to be clean evidence of expiry effects, which supports reading the expiry effect at the end of the day.
* **REMAINING ISSUES:** A finer (intraday) outcome for the sector indices would need minute bars that are not available here.
* **NEXT ITERATION:** The Bank Nifty forecasting replication (P91).

## Iteration 37: answering an AI referee on the forecasting contest

* **OBJECTIVE:** Test whether the forecasting headline (the hybrid beats mcsGARCH at every horizon) survives the referee's objections.
* **PROBLEM IDENTIFIED:** The hybrid was the best of 32 models, so its p-value ignores the search; it sees VIX and option prices while mcsGARCH sees neither; the hold-out had been covered by the first run; and plain LightGBM beat the hybrid in the hold-out.
* **HYPOTHESIS:** If the gain is real and comes from the method, it survives a data-snooping correction and an information-matched benchmark.
* **CHANGES IMPLEMENTED:** `experiments/referee_forecast.py`, `tests/test_referee_forecast.py`, `docs/REFEREE_RESPONSE_FORECAST.md`, results in `results/referee_forecast/`. Four information-matched mcsGARCH variants (VIX or straddle-implied daily level, an EGARCH-X with log VIX) on the same 38 walk-forward refits. All post hoc; nothing in the pipeline or its results changed.
* **EXPERIMENTS PERFORMED:** Hansen's SPA and White's Reality Check with step-down adjusted p-values; seed-noise-adjusted DM tests; the Model Confidence Set listed in full; LightGBM against the hybrid in the hold-out and over all days; a real-time version of mcsGARCH's fallback rule.
* **RESULTS:** (REAL DATA.) SPA rejects in all 9 cells; the hybrid survives the step-down adjustment in 8 of 9 (not the last hour on expiry days, 0.062). Against the best information-matched mcsGARCH the hybrid is 4.8% (p 0.005) and 5.9% (p below 0.001) better at 15 and 30 minutes, but only 1.7% (p 0.23) in the last hour: the last-hour gain is mostly VIX information. Plain LightGBM and the hybrid cannot be told apart over all days. Seed noise is 0.19 points. A bug-like finding: the benchmark's EWMA fallback rule looked at days after the test block in two refits; a real-time rule makes the benchmark weaker, and leaving out the 42 affected days the margins are 5.9%, 7.9% and 6.5%.
* **IMPROVEMENTS:** The forecasting claim is now "more information plus a tree correction beats our mcsGARCH at 15 and 30 minutes, also against a benchmark given the same information", not "the paper's model can be beaten at every horizon".
* **REMAINING ISSUES:** A full bootstrap over LightGBM seeds (about 6 hours on one core) was not run; the pipeline still uses the fallback rule as committed (disclosed, with the real-time check).
* **NEXT ITERATION:** The expiry-identification part of the referee report.

## Iteration 38: answering an AI referee on how expiry premia are identified

* **OBJECTIVE:** Check the expiry-day evidence (R2 and the event study) against three referee objections.
* **PROBLEM IDENTIFIED:** The week-relative baseline contains other indices' expiry days (M9); the event-study placebo p-values rest on few independent windows and the Friday control contains Sensex expiry Fridays (M8); and one-minute variance at the end of expiry days may be inflated by microstructure noise (M10).
* **HYPOTHESIS:** If the end-of-day expiry premium is real, it survives a clean baseline, honest small-sample inference and a noise-robust variance estimator.
* **CHANGES IMPLEMENTED:** `experiments/referee_expiry.py`, `tests/test_referee_expiry.py`, `docs/REFEREE_RESPONSE_EXPIRY.md`, results in `results/referee_expiry/`, and a verified NSE all-index expiry calendar from 1,933 full bhavcopies (`data/external/nse_all_index_contracts_by_expiry.csv.gz`, hashes in the matching source file). All post hoc.
* **EXPERIMENTS PERFORMED:** R2 and the event study on a baseline free of every NSE and BSE index expiry; baseline-free checks (raw two-day difference, a pooled regression with a dummy for every index's expiry); placebo inference with non-overlapping windows and a prediction-interval test; the Friday control without Sensex Fridays; realized kernel and pre-averaging estimators; pre-trends in 6- and 12-month windows.
* **RESULTS:** (REAL DATA.) The pre-November 2024 premium cannot be identified (0.10 expiry-free days per week from September 2023 to November 2024). The Tuesday premium grows on the clean baseline (+26% whole day, +63% last hour). Bank Nifty's 2023 switch survives without a baseline (+0.90) and with all-expiry dummies (+0.61). Six-month placebo p-values rest on at most 4 independent windows (prediction p 0.056 to 0.080). Expiry-day last-30-minute returns are negatively autocorrelated, and a realized kernel keeps a half to three quarters of each premium. The switch comparison is null at event d.
* **IMPROVEMENTS:** The expiry claim is now "expiry days show more, and more quickly reversing, price movement in the last hour", with the clean-baseline and noise caveats in the pitch, briefing, results summary and event study.
* **REMAINING ISSUES:** BSE expiry dates are still reported, not verified; tick data would be needed to separate noise from efficient-price variance.
* **NEXT ITERATION:** Fold all referee responses into the working paper and the replication note.

## Iteration 39: does the forecasting result replicate on Bank Nifty? (registered plan, P91)

* **OBJECTIVE:** Test whether the Nifty forecasting hybrid, every choice of which was made on Nifty, also beats mcsGARCH on Bank Nifty with nothing changed.
* **PROBLEM IDENTIFIED:** A model tuned on one index may fit that index's quirks; the anchor paper fits mcsGARCH to both indices, so Bank Nifty is the natural out-of-sample check.
* **HYPOTHESIS:** H1, the hybrid beats a 21-day-refit mcsGARCH at 15 and 30 minutes after Holm over the two; H2 to H4 (last hour, expiry days, GEX adds nothing) secondary.
* **CHANGES IMPLEMENTED:** `run_pipeline.py --symbol` and `build_market_from_nse(other_symbol=...)`; the Nifty run reproduces the committed v2 files byte for byte. Plan committed first (`docs/prereg/P91_BANKNIFTY_FORECAST.md`, commit 97eb0a1). Results in `results/pipeline_nse_banknifty/`, write-up `docs/BANKNIFTY_FORECAST.md`.
* **EXPERIMENTS PERFORMED:** The registered command, run once (78 minutes), on 660 Bank Nifty test days from 10 August 2023 to 22 April 2026.
* **RESULTS:** (REAL DATA.) Replicated: 3.9% and 5.0% better at 15 and 30 minutes (Holm p 5e-12 and 2e-11), about 70% of Nifty's gain over the same dates. Last hour 8.1% (p 3e-7). Expiry days 2.8% and 3.4% (Holm p 0.011). The GEX prediction failed: GEX features help by 0.6% to 0.8% at 15 minutes under every sign rule, the placebo does not, and all of it is at 09:15 (exploratory). No deviations from the plan.
* **IMPROVEMENTS:** The forecasting result is no longer a single-index result.
* **REMAINING ISSUES:** Bank Nifty is not independent of Nifty; the 09:15 GEX gain needs its own pre-registered test on new data.
* **NEXT ITERATION:** Rebuild the test ledger with every study.

## Iteration 40: second revision round, number tracing and one command for everything

* **OBJECTIVE:** Bring every summary document in line with the three referee responses and P91, and make the "one command reruns every study" claim true.
* **PROBLEM IDENTIFIED:** The pitch (about 2,150 words) and briefing (about 1,800) had grown with each referee round; the working paper still said the Giacomini and White test "was not used" and "GEX adds nothing" without the Bank Nifty exception; `reproduce_all.py` did not run P96, P91, the referee analyses, the ledger or the number trace; P96's 1,456 raw files were not in the hash manifest; the demo notebook asked for a model (`lgbm_gex`) that the v2 contest no longer has, so the reproduction's notebook step failed; and nobody had asked whether P90's registered DSR verdict could have come out the other way.
* **HYPOTHESIS:** None (housekeeping plus one post hoc calculation).
* **CHANGES IMPLEMENTED:** Pitch shortened to about 1,750 words and briefing to about 1,400, both with every result number in the registry (pitch: nothing uncovered; settings marked UNTRACED). Working paper revision round 2 (private, project files only): P91 paragraph, Giacomini and White results, Figure D1 from the v2 panel, the DSR probability. `reproduce_all.py` now runs every real-data study; `data/MANIFEST.csv` covers 4,446 files. Notebook fixed and re-executed on real data. README test count 259 to 488 (full suite passes; 4 tests need raw Bank Nifty data). New `experiments/f2_dsr_power.py` with 4 tests.
* **EXPERIMENTS PERFORMED:** Number trace with the private paper (286 entries); post hoc DSR power (normal approximation used by the DSR itself).
* **RESULTS:** Trace: 275 MATCH, 6 UNTRACED, 5 MISMATCH (RD30 to RD36, the R5 table, waiting for the reproduction). DSR (POST HOC, REAL DATA inputs): passing needed an annualised Sharpe above about 7.08 over 389 days; an edge of the observed size with no costs passes with probability 4e-13, and a 50% chance needs about 20 bp per trade, five times the observed 4.15 bp. The registered verdict was close to certain; the cost grid is the informative result.
* **IMPROVEMENTS:** Shorter, fully traced summaries; one command for every study; an honest reading of the P90 verdict.
* **REMAINING ISSUES:** The R5 table cells (RD30 to RD36) wait for the reproduction; the seed bootstrap (P99) is running.
* **NEXT ITERATION:** Compare the reproduction with the committed outputs.

## Iteration 41: the full reproduction and a ledger of every study

* **OBJECTIVE:** Check that the whole project reruns from the hashed raw files with one command, and count every hypothesis test of all 25 studies in one ledger.
* **PROBLEM IDENTIFIED:** The ledger (iteration 32) covered 17 studies and 4,930 tests; the registered plans P91, P92 and P96 and the three referee responses were missing. No full rerun had been logged.
* **HYPOTHESIS:** None (verification).
* **CHANGES IMPLEMENTED:** Full run of `scripts/reproduce_all.py` (22 steps, logged in `results/reproduce_log.csv`) and a file-by-file comparison with the committed outputs. Notebook cell fixed (a renamed model column). Diagnostic files from the first forecasting run that the rerun replaced with second-run versions are kept in `results/archive_v1_first_run/` and the documents that cite them point there. The R5 table in REAL_DATA_RESULTS.md had mixed 929-day and 932-day estimates; it now quotes `results/real_day_level/regressions.csv` throughout. Ledger adapters for BNFC (P91), SWAP (P92), SYNTH (P96), REFFC, REFEXP and REFF2, plus the archived first-run files. The wording that a "second person" re-keyed the paper's numbers was wrong (it was a separate AI agent) and is corrected in the code and the documents.
* **EXPERIMENTS PERFORMED:** Full rerun (about 2 hours 20 minutes on one core); ledger rebuild; number trace with the private paper.
* **RESULTS:** Every step exited cleanly except the notebook (fixed). The headline forecasts (`results/pipeline_nse_v2/`) are byte for byte the same; other studies moved only in late decimals because they now read the second run's GEX panels; P87 is identical. Ledger (REAL DATA): 22,525 tests from 25 studies, 6,595 with raw p below 0.05, 2,691 survive project-wide Benjamini-Yekutieli (cut-off about 5.6e-4 on the raw p), 1,925 survive Holm within their study. Against the findings: the hybrid's gain over mcsGARCH on all days survives at every horizon; on expiry days only at 30 minutes; the edge over the mcsGARCH given the implied variance does not survive at either horizon; P91 H1 survives, none of the 12 H4 rows does (the "prediction failed" reading rests on the plan's own Holm family); the Tuesday last-hour premium survives; the in-sample R5 row survives (BY 0.021) but the out-of-sample and referee rows do not; nothing in P92 or P96 survives, as their nulls imply.
* **IMPROVEMENTS:** The "one command" and "every test counted" claims are now both checked. The pitch, briefing and REAL_DATA_RESULTS quote the new ledger and say which parts of finding 2 do not survive.
* **REMAINING ISSUES:** The timed run predates the steps added for P96, P91 and the referee analyses (each was run on the same data when written). P99 seed bootstrap still running.
* **NEXT ITERATION:** Refresh the ledger numbers in the private working paper and replication note; score the frozen 3 November forecast after the close.

## Iteration 42: does the forecasting headline survive other LightGBM seeds? (registered plan, P99)

* **OBJECTIVE:** Close the referee's estimation-noise item (`docs/REFEREE_RESPONSE_FORECAST.md`, M4.3) with the seed bootstrap that was only costed there, not run.
* **PROBLEM IDENTIFIED:** The headline DM p-values treat one fitted hybrid as fixed; seed noise had been measured from three single-seed runs at 30 minutes only and added to the DM variance as an assumption.
* **HYPOTHESIS:** The headline holds if every one of 20 new 5-seed ensembles beats mcsGARCH with one-sided DM p below 0.05 at 15 and 30 minutes on all days (registered rule); S1 to S3 secondary (a bootstrap over days and seeds, the last hour, expiry days).
* **CHANGES IMPLEMENTED:** `experiments/seed_bootstrap.py` (reuses the pipeline's loader, `run_walkforward`, `diebold_mariano` and the SPA's stationary bootstrap; no pipeline default changed), `tests/test_seed_bootstrap.py` (12 tests, synthetic only). Plan committed and pushed first (`docs/prereg/P99_SEED_BOOTSTRAP.md`, commit 64c6151). Results in `results/seed_bootstrap/`, write-up `docs/SEED_BOOTSTRAP.md`.
* **EXPERIMENTS PERFORMED:** 63 walk-forwards (base seeds 7 and 101 to 120, three targets; 71 minutes on 2 cores) on the v2 data; the headline seed reproduces the committed forecasts bit for bit and mcsGARCH is identical in all 21 runs.
* **RESULTS:** (REAL DATA.) Holds. All 20 ensembles: 5.64% to 5.92% better at 15 minutes (largest p 1.1e-8), 7.76% to 8.11% at 30 minutes (largest p 1.8e-13); seed bootstrap p below 0.001 at both; the headline's 7.65% at 30 minutes is the smallest gain of all 21 seed sets. Last hour holds on all days (largest p 0.00043) but is weakened on expiry days (19 of 20 below 0.05, worst p 0.052, median 0.026; the headline's 0.0097 was the most favourable of 21; bootstrap p 0.023). Seed noise SD 0.075 to 0.244 points on all days, about a tenth of the DM standard error. No deviation from the plan; one post hoc comparison file (`m43_comparison.csv`).
* **IMPROVEMENTS:** The 15- and 30-minute claim no longer rests on one set of seeds; the last-hour expiry p of 0.010 should be quoted as "0.015 to 0.052 across seeds".
* **REMAINING ISSUES:** Refit dates, hyperparameters and the model choice are held fixed, so this is a lower bound on estimation noise.
* **NEXT ITERATION:** Fold the seed range into the referee response (M4.3) and the working paper.

## Iteration 43: is the forecasting gain worth anything in money? (registered plan, P103)

* **OBJECTIVE:** Answer the AI referee's remark (M5) that "no economic evaluation ... is offered" for the forecasting result.
* **PROBLEM IDENTIFIED:** A lower QLIKE loss says the hybrid forecasts better, not that anyone would pay for it; the project had no money-terms yardstick.
* **HYPOTHESIS:** For an investor who sizes a Nifty position by forecast volatility (the realized-utility yardstick of Bollerslev, Hood, Huss and Pedersen 2018), the hybrid beats mcsGARCH (H1); the size label, the information-matched comparison and a volatility-targeted trading rule after costs (with a Deflated Sharpe Ratio) are secondary, with a decision table fixed in advance.
* **CHANGES IMPLEMENTED:** `experiments/p103_economic_value.py`, `tests/test_p103_economic_value.py` (19 tests, synthetic only). Plan committed and pushed first (`docs/prereg/P103_ECONOMIC_VALUE.md`, commit 77b435a); results 13 minutes later (2e40028). Write-up `docs/P103_ECONOMIC_VALUE.md`.
* **EXPERIMENTS PERFORMED:** One run on the committed v2 forecasts (792 test days, about 5 seconds).
* **RESULTS:** (REAL DATA.) Registered verdict "detectable but not economically meaningful". H1 holds (one-sided p 8.1e-7, better on 69% of days), but the fee for switching from mcsGARCH to the hybrid is 0.041% of wealth a year (label "tiny"); against the mcsGARCH given the implied variance it is not significant after Holm (p 0.096). The volatility-targeted intraday rule loses money whatever forecast it uses (net Sharpe ratio -3.52, Deflated Sharpe Ratio 6e-15). No deviation from the registered analysis.
* **IMPROVEMENTS:** The pitch, briefing and paper now state the economic size of the forecasting gain and stop short of any trading claim.
* **REMAINING ISSUES:** One yardstick (a mean-variance investor at 30-minute rebalancing); option-based uses of the forecasts (straddle pricing, hedging) are not tested.
* **NEXT ITERATION:** Fold the result into the referee response and the working paper (done the same morning).

## Iteration 44: does the forecasting hybrid replicate on a third index? (registered plan, P104)

* **OBJECTIVE:** Test the forecasting headline on a third NSE index, with nothing changed, as the strongest available guard against the hybrid having been tuned to Nifty.
* **PROBLEM IDENTIFIED:** Two indices (Nifty, Bank Nifty P91) is a thin base for a general claim; Sensex is ruled out (BSE files cannot be reached from the cloud machine) and Midcap Select has no free index minute bars for 2023 to 2026.
* **HYPOTHESIS:** H1 (primary): the hybrid beats mcsGARCH at 15 and 30 minutes on all Fin Nifty test days (Holm over the two). H2 last hour, H3 Fin Nifty expiry days, H4 GEX features do not help (secondary).
* **CHANGES IMPLEMENTED:** Fin Nifty support in `src/niftygex/data.py` and `features.py` and in `scripts/run_pipeline.py`, `scripts/fetch_finnifty_data.py` for the pinned Fin Nifty bars (ayush-roy-21/EigenTrade, commit 21fc3e7; the repository states no licence or origin), the minute-bar check in `scripts/validate_minute_bars.py`, `experiments/p104_finnifty_summary.py`, `tests/test_symbol_finnifty.py`, and the new raw files in `data/MANIFEST.csv`. Data check first: the bars' daily high is within 5 bp of NSE's official value on 98.9% of days, the low on 99.2%. Plan committed and pushed first (`docs/prereg/P104_THIRD_INDEX_FORECAST.md`, commit a1e84a5); results e4aba76. Write-up `docs/THIRD_INDEX_FORECAST.md`.
* **EXPERIMENTS PERFORMED:** The full v2 pipeline on Fin Nifty (604 test days, 61 minutes), plus two labelled exploratory looks.
* **RESULTS:** (REAL DATA.) H1 replicated: 4.9% and 6.1% better (Holm p 4.4e-12 for both; project-wide BY 8.3e-10 and 1.2e-9), close to Nifty's 5.1% and 6.6% on the same dates. H2 not supported (1.0%, p 0.30). H3 not supported (3.9% and 5.3% on 82 expiry days, Holm p 0.093). H4 held (smallest Holm p 0.40): the Bank Nifty 09:15 GEX gain did not reappear. Post hoc: mcsGARCH's 09:15 forecast on 5 June 2024 was 135 times the realized variance and is 98% of its 15-minute squared error on Fin Nifty (79% on Nifty, 42% on Bank Nifty), so MSE comparisons against mcsGARCH mostly measure that morning; QLIKE results hardly move without it. No deviation from the registered analysis.
* **IMPROVEMENTS:** The 15- and 30-minute claim now rests on three indices under registered plans; the last-hour claim is narrowed to "depends on the index", the expiry-day claim softened, and GEX helped on one index of three.
* **REMAINING ISSUES:** Fin Nifty moves with Bank Nifty and Nifty and borrows Nifty's VIX, so this is not three independent tests; the minute bars' source is of unknown origin; the project had used Fin Nifty daily data before (P96).
* **NEXT ITERATION:** Fold into the pitch, briefing, results summary, working paper and referee response; caveat every MSE comparison against mcsGARCH.

## Iteration 45: one pooled test over the three indices (registered plan, P107)

* **OBJECTIVE:** Replace three separate replications of the forecasting hybrid with one test that respects how the indices move together, and measure how much independent information Bank Nifty and Fin Nifty add.
* **PROBLEM IDENTIFIED:** Nifty, Bank Nifty and Fin Nifty share dates, news, India VIX and the participant file, so "replicated on three indices" overstates the evidence.
* **HYPOTHESIS:** P1 (primary): pooled over the three indices, the hybrid's mean daily QLIKE is lower than mcsGARCH's at 15 and 30 minutes, with Driscoll-Kraay standard errors (Holm over two). Secondary: a joint bootstrap over dates, equal gains across indices, the last hour, expiry days, and the edge over the mcsGARCH given the implied variance. Descriptive: the correlation of daily loss differences and an effective number of independent indices. The plan says plainly that each index's own result was already known, so the pooled test is a summary, not new evidence, and shows in advance that the primary test could not fail.
* **CHANGES IMPLEMENTED:** `experiments/p107_pooled_test.py` (its own Driscoll-Kraay estimator, checked against a hand calculation and statsmodels), `experiments/p107_posthoc.py`, `tests/test_p107_pooled_test.py` (synthetic only). Plan committed and pushed first (`docs/prereg/P107_POOLED_THREE_INDEX.md`, commit eb8580a); results 9 minutes later (af0bf1a). Write-up `docs/P107_POOLED_THREE_INDEX.md`. Ledger adapter POOL3.
* **EXPERIMENTS PERFORMED:** One run on the committed forecasts of the three indices, 604 common dates (10 August 2023 to 27 January 2026).
* **RESULTS:** (REAL DATA.) P1 holds: 4.7% and 6.0% lower loss (t 10.7 and 11.3; bootstrap agrees). The three gains cannot be told apart (p 0.23 and 0.37). Daily advantages correlate 0.22 (Nifty, Bank Nifty), 0.41 (Nifty, Fin Nifty) and 0.60 (Bank Nifty, Fin Nifty) at 15 minutes, so the three indices are worth about 1.8 independent ones at 15 minutes and 2.4 at 30 minutes; added last, Fin Nifty adds almost nothing. The pooled edge over the information-matched mcsGARCH is 2.5% and 3.0% (Holm 0.043 within the study, not significant project-wide). In the last hour the indices differ (Fin Nifty about 1%). No deviation from the registered analysis.
* **IMPROVEMENTS:** "Replicated on three indices" becomes "replicated on three related indices, worth about two independent replications".
* **REMAINING ISSUES:** Effective numbers are noisy and sensitive to a few extreme days (4 June 2024); equal weights are a choice.
* **NEXT ITERATION:** Fold into the paper and the pitch.

## Iteration 46: a blind re-derivation of the paper's "before" window (registered plan, P108)

* **OBJECTIVE:** Check, independently, the project's most important single claim about the paper: that its "before" sample is not the stated November 2023 to February 2024 but a 64-day window starting in mid-November 2022.
* **PROBLEM IDENTIFIED:** The earlier search (iteration 17) was designed by the same process that then believed its answer, and scored only four statistics. A finding that the professor will be asked about directly should survive a fresh attempt that cannot see it.
* **HYPOTHESIS:** A separate AI agent, which types the printed numbers itself, writes new code and may not open the earlier search until its own answer is pushed, will land on the same window. The plan (`docs/prereg/P108_BLIND_WINDOW.md`, commit 6b289ca) fixes the candidate return definitions, the score (exact matches among 15 printed numbers first, then distance), and what counts as a strong or weak identification.
* **CHANGES IMPLEMENTED:** `experiments/p108_blind_window/` (`printed_values.csv`, `search.py`, `posthoc.py`, `phase3_compare.py`), `tests/test_p108_blind_window.py` (synthetic only), write-up `docs/P108_BLIND_WINDOW.md`. Blind answer pushed at d904c8c, 21 minutes after the plan; comparison with the earlier search afterwards (73ca9d3).
* **EXPERIMENTS PERFORMED:** 132,126 windows under 204 return definitions; POST HOC checks of the minima, the after window and fixed kurtosis formulas.
* **RESULTS:** (REAL DATA.) The stated window reproduces none of the 15 printed numbers under any definition (rank 85,013). The blind best fit is 22 November 2022 to 20 February 2023, 6 of 15 exact (both N, both means, both maxima); the window one day later ties, so under the plan this is a weak identification of the exact days. Return definition, year, length and the minutes of both maxima agree with the earlier search; the minima mismatch has the same cause (the 27 January 2023 15:15 bar). The earlier answer, 14 November 2022 to 10 February 2023, ranks 9th: its Bank Nifty mean is -0.00000055 and prints as -0.000001, not -0.000002 (checked again by the lead from the raw bars: 22 November gives -0.0000018). The higher moments, with the kurtosis formula fixed, favour starts of 14 to 17 November.
* **IMPROVEMENTS:** "The before window is 14 November 2022 to 10 February 2023" becomes "the before window starts between 14 and 23 November 2022 and ends between 10 and 21 February 2023", with the year confirmed by two independent searches.
* **REMAINING ISSUES:** The exact days depend on whether the printed means or the printed higher moments are more trustworthy; only the authors can settle it. Analyses that use the inferred window use 14 November to 10 February; the two candidates share 58 of 64 days. No hypothesis tests, so nothing enters the test ledger.
* **NEXT ITERATION:** Soften the window claim in the pitch, briefing, paper, replication note and deck; ask the authors which dates they used.

## Iteration 47: where the forecasting hybrid's gain comes from (registered plan, P106)

* **OBJECTIVE:** Say what the hybrid's 5.8% and 7.6% gain over mcsGARCH is made of, and why its edge over the information-matched mcsGARCH appears only after SEBI's November 2024 cut.
* **PROBLEM IDENTIFIED:** The summaries said "much of the gain is extra information rather than a better method", from the fact that an mcsGARCH given the implied variance came close before November 2024. That is a comparison of two models, not a measurement of what the hybrid itself uses. LightGBM's split-gain importance ranked the option-implied inputs first, which does not show what the hybrid would lose without them.
* **HYPOTHESIS:** Registered before any ablated loss was computed (`docs/prereg/P106_GAIN_DECOMPOSITION.md`, commit 71bb3b6): refitting the hybrid without each of five input groups (option-implied, VIX, time of day, calendar, realized history), with five seed sets on Nifty, raises its loss for the groups that matter (P1, Holm over 20 primary rows); P2 asks whether any group's contribution changed across the cut. Secondary: benchmark or hybrid as the source of the period change, blocks of groups, the last hour, Bank Nifty and Fin Nifty.
* **CHANGES IMPLEMENTED:** `experiments/p106_gain_decomposition.py`, `WalkForwardConfig.drop_sets` in `src/niftygex/forecast.py` (empty by default; the headline forecasts reproduce bit for bit), `tests/test_p106_gain_decomposition.py` (synthetic). Results 149b358, 119 minutes after the plan. Ledger adapter GAIN6.
* **EXPERIMENTS PERFORMED:** 22 Nifty refits plus Bank Nifty and Fin Nifty (about 1 hour 45 minutes on 2 cores); 189 tests.
* **RESULTS:** (REAL DATA.) Realized history carries 34% and 38% of the gain at 15 and 30 minutes, time of day 23% and 25% (Holm below 0.01, five of five seed sets). The option-implied, VIX and calendar groups add nothing (calendar slightly hurts). A hybrid given only mcsGARCH's own inputs is 6.06% and 8.01% better than mcsGARCH. Same leaders on Bank Nifty and Fin Nifty; realized history carries 69% of the last-hour gain. Across the cut the hybrid's own gain was steady (p 0.45 and 0.29), while the information-matched mcsGARCH went from 4.98 and 6.06 points better than mcsGARCH to 1.06 and 0.64 worse (87% and 82% of the change in the edge; its own Holm p 0.135). No P2 row passes (implied group at 30 minutes, Holm 0.051; a seed-only placebo shows the period test can overstate). One deviation: the P99 reproduction check compared text with datetime dates; fixed, and every other output was byte-identical.
* **IMPROVEMENTS:** "Mostly information" becomes "a learnt correction of mcsGARCH from its own kind of information". "The last-hour gain is mostly VIX information" is withdrawn: an mcsGARCH given VIX comes within 1.7% in the last hour, but the hybrid does not need VIX for its gain. The November 2024 edge is "the benchmark lost its advantage", not "the hybrid got better".
* **REMAINING ISSUES:** Leave-one-out misses what groups share (the five shares sum to about half). Part 2 uses the data that motivated it. Dropping the calendar inputs would be a new model chosen after seeing these data.
* **NEXT ITERATION:** Fold into the pitch, briefing, paper and deck.

## Iteration 48: a one-day demo of the forecasts for the meeting (P111)

* **OBJECTIVE:** Make the forecasting result concrete for one trading day, in a form that can be shown in a meeting.
* **PROBLEM IDENTIFIED:** The headline (5.8% and 7.6% lower QLIKE over 792 days) and P106's attribution are averages; a professor will want to see what the two forecasts look like on a real day.
* **HYPOTHESIS:** None tested; this is a presentation tool. Days are chosen by stated rules, not by hand.
* **CHANGES IMPLEMENTED:** `scripts/demo_forecast_day.py` (reads committed forecasts only, writes one self-contained HTML page, about a second per day), `docs/DEMO.md` (walk-through), `tests/test_demo_forecast_day.py` (synthetic), three pages and a deck chart in `results/demo/`. Built by an AI agent (Claude); commit 59dc9c4.
* **EXPERIMENTS PERFORMED:** Three days by rule: the median-gain Tuesday expiry since October 2025 (23 December 2025), the 4 June 2024 election-result crash (named in advance), and the median 09:15 win among non-expiry days (8 April 2025).
* **RESULTS:** (REAL DATA, 15-minute forecasts.) 23 December 2025: QLIKE 0.166 for mcsGARCH and 0.155 for the hybrid, nearly all of the difference in the 09:15 window (0.850 against 0.395); the hybrid is closer in only 8 of 25 windows (checked again by the lead from the committed forecasts). 4 June 2024: the hybrid is 2.5% worse at 15 minutes. 8 April 2025: 13.7% better.
* **IMPROVEMENTS:** A slide with the 23 December chart is in the meeting deck.
* **REMAINING ISSUES:** One day's last-hour row rests on a single forecast and can look extreme; `docs/DEMO.md` says so.
* **NEXT ITERATION:** None planned.

## Iteration 49: what the hybrid's correction fixes in mcsGARCH (registered plan, P109) and its forward test (P112)

* **OBJECTIVE:** Say in plain words what the LightGBM correction repairs in mcsGARCH, now that P106 showed it learns from mcsGARCH's own kind of information.
* **PROBLEM IDENTIFIED:** The summaries described the hybrid's 5.8% and 7.6% gain as something machine learning finds. If a few plain fixes to mcsGARCH did as well, the gain would be over the paper's mcsGARCH as specified, not over the best econometric model, and the professor should hear that from us.
* **HYPOTHESIS:** Registered before any extended model was run on real data (`docs/prereg/P109_MCSGARCH_EXTENSIONS.md`, commit ceec2ec): five fixes, each aimed at one suspected mistake (X1 HAR daily level, X2 overnight fade, X3 today's running level, X4 leverage, X5 one calibration factor per forecast time), and their combination, beat mcsGARCH at 15 and 30 minutes; the plan expected the combination to recover part of the hybrid's gain, not all of it.
* **CHANGES IMPLEMENTED:** `McsGarchExt` and helpers appended to `src/niftygex/volatility.py`, `experiments/p109_mcsgarch_extensions.py`, `tests/test_p109_mcsgarch_extensions.py` (synthetic). Results 7f0b5c7, 27 minutes after the plan, run by an AI agent. Ledger adapter MCSX (100 tests). Forward-test plan `docs/prereg/P112_FORWARD_MCSGARCH_FIXES.md` committed (f081138) before any 4 November data existed.
* **EXPERIMENTS PERFORMED:** The committed walk-forward on Nifty (792 test days), Bank Nifty (660) and Fin Nifty (604), about 16 minutes on 2 cores. The lead recomputed the two primary gains from the daily-loss files and reran the Nifty combination from scratch in a clean folder: every forecast matched the agent's to the last digit (gains 6.8305 and 8.2516 points).
* **RESULTS:** (REAL DATA.) The combination is 6.83 and 8.25 points better than mcsGARCH (one-sided DM t 5.61 and 5.01), 118% and 108% of the hybrid's gain (intervals 85% to 167% and 80% to 132%); the hybrid does not beat it (p 0.823 and 0.729). Alone, today's level recovers 46% and 48%, the time-of-day factor 49% and 39%, the overnight fade 12% and 16%, the HAR level 19% and 3% and leverage nothing. At 09:15 the combination gains 28.7 and 34.4 points against the hybrid's 13.3 and 23.9; at other times it recovers 93%. Last hour: 13% only. Bank Nifty 127% and 136%, Fin Nifty 97% and 103%, nothing retuned.
* **IMPROVEMENTS:** The pitch, briefing, README, results summary, paper (abstract, contribution 2, Section 6.2, 5.8, conclusion, Tables B1 and C), referee response and deck (new slide "Five fixes to your model") now say that the hybrid's gain is over mcsGARCH as specified, and what the correction does. Ledger: 27,988 tests, 31 studies, 3,271 survive; trace 507 entries, 501 MATCH, no mismatch.
* **REMAINING ISSUES:** The fixes were chosen after P106 on the same days, so this explains the gain rather than proving a better model; the time-of-day factor is fitted, not structural; the intervals are wide; the last hour is not explained.
* **NEXT ITERATION:** P112 runs once, at the 250th valid Nifty day from 4 November 2026 or 31 December 2027. P110 (P106's claims stress-tested on Bank Nifty and Fin Nifty) is running.

## Iteration 50: which connectedness settings reproduce the paper's hour-level numbers (registered plan, P113)

* **OBJECTIVE:** Turn the one replication gap left to "unreported settings", the hour-level connectedness totals of Tables 3 and 4, into a precise question for the authors.
* **PROBLEM IDENTIFIED:** The replication's 40-setting grid could not reach the before first-hour and last-hour totals (99.42 and 105.22). A total above 100 needs both cross shares above one half, which a row-normalised decomposition almost never gives, so either the paper used another normalisation or setting, or the numbers cannot be produced from these data.
* **HYPOTHESIS:** Registered before any connectedness number was computed (`docs/prereg/P113_CONNECTEDNESS_SETTINGS.md`, commit 313a3ac): some setting among 5,440 per before window (row, column or no normalisation, Cholesky in both orders, 8 inputs, 8 horizons, 17 constructions and lag rules) reproduces the 15 printed totals, judged by a mean-absolute-miss rule with shuffle and leave-one-table-out guards.
* **CHANGES IMPLEMENTED:** `experiments/p113_connectedness_settings.py`, optional decompositions in `src/niftygex/connectedness.py` (defaults unchanged; the replication's totals reproduce exactly), `tests/test_p113_connectedness_settings.py` (17 synthetic tests). Run by an AI agent; results c764984, 19 minutes after the plan. Ledger adapter CONNSET.
* **EXPERIMENTS PERFORMED:** 16,320 settings over three before windows, 1,000 shuffles, 999 day-level bootstrap draws; 148 seconds on one core. The lead recomputed the best fit and the before last-hour maximum from `grid_cells.csv.gz`.
* **RESULTS:** (REAL DATA.) Not reproduced. The closest setting (column-normalised generalized decomposition of the log mcsGARCH variance, days joined, VAR(5), horizon 5) misses the 15 totals by 5.34 points on average (8 within 2 points; better than shuffled totals, p 0.003) but misses the before last hour by 23.1 points; no setting whose shares sum to 100 per row exceeds 94.66 there. The after totals are fitted no better than chance (p 0.63). Under the closest setting the falls on expiry days hold (Holm p 0.012); the larger last-hour fall does not.
* **IMPROVEMENTS:** The question for the authors becomes specific (how were the shares normalised, with which input, lag and horizon). Pitch, briefing, README, results summary, paper (6.1.3, contribution 1, 5.8, Tables B1 and C), replication note, referee response and deck updated. Ledger: 28,001 tests, 32 studies, 3,271 survive; trace 536 entries, 529 MATCH, no mismatch.
* **REMAINING ISSUES:** The grid is finite (other hour definitions, time-varying VARs and CPP's exact mcsGARCH estimates were not tried); each one-hour cell has 180 observations from three days.
* **NEXT ITERATION:** None planned; the answer has to come from the authors.

## Iteration 51: an independent audit of the documents written for the professor

* **OBJECTIVE:** Check, before the meeting, that the pitch, the briefing, the private working paper and the private replication note say nothing about the paper or its authors that the code and the results files do not support.
* **PROBLEM IDENTIFIED:** Each document had been edited many times as studies landed, so wording could have drifted from the evidence.
* **HYPOTHESIS:** None (a review, no tests).
* **CHANGES IMPLEMENTED:** A separate AI agent read the four documents against the code, the results files and the paper's PDF and raised 14 points; the lead verified each against the code or the results before accepting it. The main ones: (1) our forecasting mcsGARCH is not exactly the paper's model (it adds a separate, smoothed diurnal curve for expiry days, `McsGarch(expiry_diurnal=True, smooth=5)` in `src/niftygex/forecast.py`, uses a zero intraday mean and has an EWMA fallback), so every "your model as printed" or "mcsGARCH as specified" now reads "our implementation of mcsGARCH", and the paper's Section 5.2 discloses the differences; (2) the edge over an mcsGARCH given VIX or implied variance is no longer called a gain over "the same information" (it is post hoc, concentrated since November 2024 and does not survive the project-wide correction); (3) the connectedness fall replicates only on the inferred window (on the stated window, all days -2.0, interval -5.8 to 1.5; `results/replication_nse/tables34_changes.csv`), which the pitch, the paper and the note now say; (4) P113's reach result is stated for normalised and Cholesky settings only, since unnormalised settings pass 100; (5) "mainly because the benchmark got worse" now says "mostly, on point estimates", since that change alone is not significant (Holm 0.135); (6) the HAR daily level is named as helping inside P109's combination; (7) P109's survival of the project-wide correction is caveated (its fixes were chosen after P106 on the same days); (8) the briefing's status and time stamp were stale and P113 was missing from its Done list; (9) the power claim in the briefing is labelled as a simulation and split into the GEX tests (5 to 11 years) and the Tuesday premia (0.8 to 4.1 years); and smaller points on attribution of the blind search to an AI agent, an ambiguous "it" and the paper's description of CPP's intended window (CPP, p. 5).
* **EXPERIMENTS PERFORMED:** None new. The number trace and the ledger were rerun.
* **RESULTS:** All 14 points accepted. Pitch, briefing, README, results summary, research strategy, paper, replication note, referee response and deck updated. Trace: 538 entries, 532 MATCH, 6 UNTRACED, no mismatch, every pitch number covered.
* **IMPROVEMENTS:** The documents no longer attribute a weakness of our implementation to the paper's model, and no longer overstate two results (the information-matched edge and the connectedness fall).
* **REMAINING ISSUES:** The audit was done by an AI agent, not a person; the professor's reading is the real test.
* **NEXT ITERATION:** None planned.

## Iteration 52: does the paper's fall in intraday persistence survive the fixed mcsGARCH? (registered plan, P114)

* **OBJECTIVE:** Check whether the paper's finding that intraday volatility shocks die out faster after the expiry change is a property of the market or of mcsGARCH's form.
* **PROBLEM IDENTIFIED:** P109 found that mcsGARCH's intraday GARCH has no moving level inside the day. Such a model can show alpha + beta near 1 because the day's level drifts, not because single shocks are long-lived; if the before window drifted more, part of the fall could be an artefact.
* **HYPOTHESIS:** Registered before any extended model was estimated on these windows (`docs/prereg/P114_PERSISTENCE_FIXED_MODEL.md`, commit 86e3866): under the model with P109's fixes X1 to X3, Nifty's persistence (the half-life of the impulse response to one shock at 12:15) still falls from the inferred before window to the after window, judged by the replication's day-block bootstrap (400 draws, one-sided).
* **CHANGES IMPLEMENTED:** `experiments/p114_persistence_fixed_model.py`, impulse-response helpers in `src/niftygex/volatility.py` (defaults unchanged; Table 2 and the replication's bootstrap reproduce exactly), 18 synthetic tests, one post hoc descriptive script. Run by an AI agent; results 7c1fc13, 49 minutes after the plan. Ledger adapter PERSFIX.
* **EXPERIMENTS PERFORMED:** Fits on the inferred, stated and P108 before windows and the after window, for Nifty and Bank Nifty; likelihood-ratio tests; 400 bootstrap draws; 44 rolling Nifty windows and 38 Bank Nifty windows. About 36 minutes on one core.
* **RESULTS:** (REAL DATA.) Primary verdict "survives": for Nifty on the inferred window the half-life falls from 141 to 18 minutes (reference model 359 to 72), change in equivalent persistence -0.033 (interval -0.058 to -0.003, one-sided p 0.022), with a very uncertain kept share (1.29, interval -0.04 to 6.91). Not confirmed for Bank Nifty (p 0.105) or on P108's window; on the stated window no fall for Nifty (19.6 to 17.9 minutes) and a rise for Bank Nifty. The inferred before window is the most persistent of the 44 rolling Nifty windows (median 25 minutes). The primary test does not survive the project-wide correction (1.0).
* **IMPROVEMENTS:** The persistence finding is now stated narrowly: Nifty, inferred window, cause open. Pitch, briefing, results summary, replication document, paper (6.1.2, contribution 1, 5.8, Tables B1 and C, Appendix B), replication note, referee response and deck updated. Ledger: 28,021 tests, 33 studies, 3,277 survive; trace 545 entries, 539 MATCH, no mismatch.
* **REMAINING ISSUES:** Windows of 64 and 59 days; in-sample estimates; the fixes were chosen for forecasting, not for this question; telling an effect of the expiry change from a persistent late-2022 period needs comparison windows with no rule change.
* **NEXT ITERATION:** None planned before the meeting.

## Iteration 53: do P106's three claims about where the forecasting gain comes from hold on Bank Nifty and Fin Nifty? (registered plan, P110)

* **OBJECTIVE:** Test P106's three claims (the gain comes from realized volatility and the time of day; the option-implied, VIX and calendar inputs add nothing; a hybrid that sees only mcsGARCH's kind of information gets the whole gain) on the two other indices, with more than one set of random seeds.
* **PROBLEM IDENTIFIED:** On Bank Nifty and Fin Nifty P106 had used one seed set, never ran the `mcsinfo` hybrid, and supported "adds nothing" only with tests that failed to find an effect, which is not evidence of no effect.
* **HYPOTHESIS:** Registered before any outcome was computed (`docs/prereg/P110_GAIN_STRESS_TEST.md`, commit 5128e53): with four new seed sets (101 to 104), claim 1 holds (Holm), each of the three other groups is equivalent to zero within 1 point of mcsGARCH's loss (two one-sided tests), and the `mcsinfo` hybrid beats mcsGARCH and is at most 1 point worse than the full hybrid.
* **CHANGES IMPLEMENTED:** `experiments/p110_gain_stress_test.py` and 12 synthetic tests (`tests/test_p110_gain_stress_test.py`). Run by an AI agent; results f98077f, 167 minutes after the plan. Ledger adapter GAINSTRESS.
* **EXPERIMENTS PERFORMED:** Bank Nifty (660 test days) and Fin Nifty (604) refitted with seven variants and five seed sets; 28 primary rows; pooled tests over three indices; robustness, placebo (seed sets against each other) and 28 post hoc period rows (added after outcomes were seen, so labelled post hoc). About 2.5 hours on one core.
* **RESULTS:** (REAL DATA.) Claim 1 survives on both indices: removing realized history raises the loss by 1.73 to 2.16 points and removing the time of day by 1.16 to 1.50 (Holm below 0.001, all seed sets agree). Claim 2 is qualified: the option-implied and VIX groups are equivalent to zero within 1 point (the option-implied group at 30 minutes only just, Holm 0.035), but the calendar inputs help a little (0.27 points on Bank Nifty at 30 minutes, 0.14 on Fin Nifty at 15). Claim 3 partly survives: `mcsinfo` beats mcsGARCH everywhere (4.00% to 5.53%) but is shown non-inferior to the full hybrid only on Bank Nifty at 15 minutes; on Fin Nifty it trails by 0.65 and 0.50 points. Seed set 7 was the unluckiest of five on Bank Nifty. Project-wide, 22 of the 28 primary rows survive.
* **IMPROVEMENTS:** "Option-market information adds nothing" is now stated as "worth less than about 1 point on every index"; "gets the whole gain" is kept for Nifty only. Pitch, briefing, README, results summary, paper (5.2, 5.8, Tables B1 and C, Appendix B) and referee response updated.
* **REMAINING ISSUES:** No new days (only data after these test periods can confirm the results); the 1-point margin is a choice; the three indices are worth about two independent ones (P107).
* **NEXT ITERATION:** None planned before the meeting.

## Iteration 54: how large is the gain over mcsGARCH exactly as the paper specifies it? (registered plan, P116)

* **OBJECTIVE:** Measure the hybrid's gain against mcsGARCH as Chhimwal, Pandey and Pandey specify it, not only against our implementation.
* **PROBLEM IDENTIFIED:** The audit found that our mcsGARCH adds three choices the paper does not make (a separate time-of-day curve for expiry days, a 5-minute smoothing of the curves, a zero expected return). If these made our benchmark weaker, the headline 5.8% and 7.6% would overstate the gain over the paper's model.
* **HYPOTHESIS:** Registered before any new benchmark forecast was made on real data (`docs/prereg/P116_BENCHMARK_AS_PRINTED.md`, commit 19c5486): H1, the hybrid beats `mcs_cpp` (one curve, no smoothing, ARMA(1,1) mean, Student t) at 15 and 30 minutes; H2, our implementation and `mcs_cpp` differ; Holm over four; a decision rule (understated, overstated, no material difference within 0.5 point, inconclusive) fixed in advance; every mix of the three choices (2 x 2 x 2) as a secondary.
* **CHANGES IMPLEMENTED:** `experiments/p116_benchmark_as_printed.py`, ARMA(1,1) mean helpers at the end of `src/niftygex/volatility.py` (defaults unchanged), 10 synthetic tests, one post hoc descriptive script. Run by an AI agent; results cf6c9d4, 25 minutes after the plan. Ledger adapter BENCHCPP.
* **EXPERIMENTS PERFORMED:** The same walk-forward, days, anchors and targets as the headline, on Nifty, Bank Nifty and Fin Nifty; 77 tests (4 primary, 16 secondary, 57 exploratory). About 8 minutes on two processes.
* **RESULTS:** (REAL DATA.) Against `mcs_cpp` the hybrid is 4.43% better at 15 minutes and 6.64% at 30 (Holm below 0.001) and 6.16% in the last hour. Our implementation is worse than the paper's specification on Nifty (by 1.45 and 1.09 points, Holm 0.005 and 0.014), so the registered verdict is "headline overstated" (Delta -1.37 [-2.15, -0.58] and -1.00 [-1.77, -0.23]); the expiry-day curve and the smoothing share the blame and the ARMA mean changes nothing. On Bank Nifty the headline is understated (4.73% and 6.20% against `mcs_cpp`), on Fin Nifty the verdict is inconclusive. P109's combination is 5.48% and 7.25% better than `mcs_cpp`. Project-wide, H1 survives (3.5e-8 and 9.1e-14) but H2 does not (0.18 and 0.73).
* **IMPROVEMENTS:** Every document now leads with the gain over the paper's specification (4.4% and 6.6%) and gives 5.8% and 7.6% as the gain over our own version. Pitch, briefing, README, results summary, paper (abstract, contribution 2, 5.2, 5.8, Tables B1 and C, Appendix B), referee response, deck and registry updated. Ledger with P110 and P116: 28,414 tests, 35 studies, 3,514 survive (cut-off about 5.7e-4), which moved every printed project-wide adjusted p by 5% to 10% (no verdict changed); trace 559 entries, 552 MATCH, no mismatch.
* **REMAINING ISSUES:** The studies run before this one (P106, P107, P109, P110, P114) use our implementation as the benchmark; against the paper's specification their Nifty gains over mcsGARCH would be about 1.0 to 1.4 points smaller.
* **NEXT ITERATION:** None planned before the meeting.

## Iteration 55: a second independent audit, after P116

* **OBJECTIVE:** Check again, after P116 changed the headline benchmark, that nothing the professor will read says more than the code and the results files support.
* **PROBLEM IDENTIFIED:** P116 showed that the "mcsGARCH" used by most studies is our implementation, not the paper's specification, and that it is slightly weaker on Nifty. Every sentence written before P116 that says "the paper's model" could now be wrong.
* **HYPOTHESIS:** None (a review, no tests).
* **CHANGES IMPLEMENTED:** A separate AI agent read the pitch, briefing, deck, README, results summaries, evidence documents, working paper, replication note and referee response against the code, the results files and the paper (26 numbered points: 7 high, 19 medium, plus wording). The lead checked each point before acting. The main fixes: (1) the private working paper no longer reads as if the professor had taken part in the analysis: it now says the work was done for an internship project the professor supervises, that they have not reviewed the draft or confirmed any finding, and that the replication is therefore not independent of the paper's authors; (2) every place that credits a result to "the paper's mcsGARCH" when it uses our version now says "our version", and the headline leads with the gain over the paper's specification (4.4% and 6.6%, P116); (3) P112's forward test was amended before any forward data existed (commit 76da563) to add `mcs_cpp` as a second benchmark with its own primary test (H1b), so the forward test can say something about the paper's model; (4) the claim that one command reruns every study was false, so `scripts/reproduce_all.py` now has a step for every registered study, a `--list` mode and a test that fails if a registered plan has neither a step nor a pending entry (7786930); (5) "both cross shares above one half" was a wrong necessity claim and is gone; (6) post hoc comparisons are labelled post hoc, "pinning is real" is stated with its correction, and the pitch now says that plain LightGBM is as good as the hybrid (p 0.32 to 0.97); (7) the one-day demo chart and pages label the benchmark "mcsGARCH (our version)" (6768b46); (8) evidence documents P103 to P110 carry a note saying which mcsGARCH they use (048ee5a). One point was not acted on in the repository: the repository is public and already states that the paper's window does not replicate. Changing that is the student's decision, so it is flagged to them, not changed.
* **EXPERIMENTS PERFORMED:** None new. The trace, the ledger and the demo were rerun.
* **RESULTS:** All high and medium points accepted except the visibility point (flagged). Pitch, briefing, README, results summary, evidence documents, paper, replication note, referee response, demo and deck (13 versions so far) updated. Trace: 575 entries, 562 MATCH, 13 UNTRACED, no mismatch, every pitch number covered. Ledger unchanged in size (28,414 tests, 35 studies, 3,514 survive).
* **IMPROVEMENTS:** No document now credits a weakness or a strength of our implementation to the paper's model, and the only remaining claim about the paper's model (the 4.4% and 6.6% gain) is backed by P116's registered test.
* **REMAINING ISSUES:** Both audits were done by AI agents. The full timed rerun of every step has not been done end to end (the steps were timed separately; stated total about 11.4 hours).
* **NEXT ITERATION:** None planned before the meeting.

## Iteration 56: is the persistence fall larger than in windows without the expiry change? (registered plan, P117)

* **OBJECTIVE:** Decide whether the paper's fall in intraday persistence is evidence that the expiry change made shocks die out faster.
* **PROBLEM IDENTIFIED:** P114 found that the fall survives a better model on the inferred window, but also that this before window is the most persistent of 44 rolling Nifty windows. A fall that large could be an ordinary return from an unusually persistent period.
* **HYPOTHESIS:** Registered before any placebo window, Fin Nifty model or test statistic was computed on real data (`docs/prereg/P117_PERSISTENCE_PLACEBO.md`, commit 27d39ba): the actual change in the log half-life of a 12:15 shock is lower (a larger fall) than the changes between placebo pairs of windows with the same lengths (64 and 59 days) and gap (258 trading days) that straddle neither Bank Nifty move; primary for Nifty under the reference mcsGARCH and P114's fixed model F, Holm over two; a decision rule (unusual, unclear, within the normal range, no verdict) fixed in advance; secondaries for Bank Nifty, other placebo sets, a conditional placebo and a difference-in-differences against Fin Nifty, whose expiry calendar did not change.
* **CHANGES IMPLEMENTED:** `experiments/p117_persistence_placebo.py`, 1 post hoc descriptive script, synthetic tests. Run by an AI agent; results d584e4f, 76 minutes after the plan. Ledger adapter PERSPLAC; `scripts/reproduce_all.py` gained its six steps.
* **EXPERIMENTS PERFORMED:** 10,524 window fits (5,404 for Nifty and 5,120 for Bank Nifty, both models) from 2018 to 2026, Fin Nifty fits with 400 bootstrap draws per key, 26 tests. About 61 minutes on one core.
* **RESULTS:** (REAL DATA.) Verdict "the fall is within the normal range" under both models: for Nifty, 233 of 1,223 placebo pairs fall at least as far under the reference (359 to 72 minutes, p 0.191) and 128 under F (141 to 18 minutes, p 0.105), Holm 0.211; Bank Nifty the same (Holm 0.193). Under F the before window is at the 94.8th percentile of placebo before windows and the after window at the 19.4th. Given the before window, the after window is ordinary (S5) except for Bank Nifty under the reference model only. Fin Nifty fell as much (72 to 9 minutes under F), so no difference-in-differences row is "larger than the control's" (Holm 1.000). Post hoc: a flat-bar NSE outage day (24 February 2021) and pairs after March 2024 set much of Nifty's placebo range; Bank Nifty had falls as large in calendar-stable years. No row survives the project-wide correction.
* **IMPROVEMENTS:** The documents now say that the persistence fall gives no evidence that the expiry change made shocks die out faster. Pitch, briefing, results summary, registry, paper, replication note and referee response updated. Ledger: 28,440 tests, 36 studies, 3,514 survive; trace 600 entries, 587 MATCH, no mismatch.
* **REMAINING ISSUES:** Only three or four separate periods stand behind the placebo range, so the test has little power either way; one event; the reference model's half-life hits its bound in many windows.
* **NEXT ITERATION:** None planned before the meeting.

## Iterations in progress (3 November 2026, about 18:45 IST)

* **Scoring the frozen forecast** for 3 November after the minute bars are published.
* **A meeting deck** for Prof. Pandey (private artifact, 14 slides), refreshed last from the final numbers.
* **P118**, the identical hybrid built on mcsGARCH as the paper specifies it (plan 49d19ce, running).
* **A timed rerun of every reproduce step** at commit 0de679b, in a separate copy of the repository (started 12:26 UTC).

Each will be written up in the same format when it finishes. The clean-clone audit of a3a84c7 (512 tests: 508 passed, 4 skipped) is in `docs/REPRODUCIBILITY.md`; the methods references (22 with DOIs) are in `docs/LITERATURE_REVIEW.md`, section 8.
