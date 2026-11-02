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

## Iterations in progress (3 November 2026, early morning UTC)

* **Intraday gamma at the current price:** the first pass is written up in `docs/INTRADAY_GAMMA.md` (a clean null on expiry days); a follow-up on the final half hour before the auction and a pre-registered Bank Nifty test is running.
* **Stronger forecasting models:** option-implied features, new GEX measures, a Student t Diebold-Mariano test and more frequent mcsGARCH refits in the forecasting contest.

Each will be written up in the same format when it finishes.
