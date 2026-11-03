# Defence summary for Prof. Pandey: ten questions, plain answers

Prepared 3 November 2026 by validation agent 2 (an AI agent). Every number is REAL (NSE data), SYNTHETIC (simulated), PAPER (printed in your 2025 paper) or OUR CHECK (recomputed during validation), and every one is consistent with the project's latest verified results. Files named are in this repository.

**Five-line version.** (1) We rebuilt your volatility model and asked whether option-dealer positions (GEX) help forecast Nifty's next 15 to 60 minutes. (2) They do not. (3) A machine-learning correction to your model does forecast better at 15 and 30 minutes (4.4% and 6.6% lower QLIKE loss than your model as you specify it, REAL), but the gain comes from recent volatility and time of day, and five plain fixes to the model recover all of it. (4) Your paper replicates closely except the stated "before" window. (5) Nearly everything beyond the forecasting result is suggestive or in-sample only, and all reviews so far were done by AI agents, not people.

## 1. How do you know this is correct?

Three layers. First, the code was tested against simple truths: Black-76 gamma against a second finite difference (agree to 6 digits), GEX against its hand formula, QLIKE against its minimiser, the connectedness index against a separate calculation (OUR CHECK, `reports/checks/equation_vs_code_check.py`, 32 of 32 checks pass). The package has 715 tests (708 pass; 7 need raw data). Second, I recomputed the headline gains directly from the committed forecast files: 5.80%, 7.65% and 6.36% against our mcsGARCH, and 4.43%, 6.64% and 6.16% against your specification, over 792 test days, matching the README (OUR CHECK). Third, a script traces 678 numbers in the write-ups to result files: 664 match, none mismatch, 14 are design constants marked untraced. The checks called independent were done by separate AI agents; no human has reviewed the work.

## 2. Why this evaluation?

QLIKE is a standard score for volatility forecasts that stays valid even though true variance is only seen through noisy realized variance (Patton 2011). It is scale-free, so calm and wild days count equally, and it punishes forecasts that are too low. Tests use daily average losses, so n is 792 days, not tens of thousands of minutes; we showed the minute-level tests in your Tables 5 and 6 treat dependent minutes as independent (p-values as small as 10^-256). The Diebold-Mariano test I re-implemented separately rejects a true null 5.2% of the time at the nominal 5% (5.6% with autocorrelated differences; SYNTHETIC, OUR CHECK). It is a forecasting question, not a trading one, so no Sharpe ratios.

## 3. Could it be overfitting?

It was the main worry, so there are five defences. Forecasts are walk-forward: the first 150 days are training only, models are refitted every 21 days, hyperparameters are chosen on the last 20% of each training window, and day t uses only data published by the close of day t-1. The gain survives a correction for having tried 31 models (Hansen SPA), 20 new sets of random seeds, and a later hold-out. It replicates when run unchanged on Bank Nifty (3.9% and 5.0%) and Fin Nifty (4.9% and 6.1%), though the three indices count as about two independent replications. Of the project's 28,515 tests only 3,544 survive the project-wide correction, and the pitch says so. The weak spot: the five fixes to mcsGARCH were chosen after seeing these days. A forward test on Nifty days from 4 November 2026 is registered (`docs/prereg/P112_FORWARD_MCSGARCH_FIXES.md`); it is not yet reported.

## 4. What assumptions matter most?

(a) The sign of dealer positions. Public data never show who holds what, so GEX needs an assumed rule. We report three (US convention, NSE "Pro" participants, mirror of "Client") and none helps forecasts; estimating the sign from the market's response is inconclusive. (b) Realized variance of one-minute returns is a fair target; at the close, bid-ask bounce on expiry days contaminates it. (c) Black-76 with a flat 6.5% rate is adequate for weekly options. (d) The 2023 to 2026 sample is representative; NSE rule changes (the closing auction from 3 August 2026) already shifted one result. Section 2 of `reports/METHODOLOGY_VALIDATION.md` lists the rest.

## 5. What if conditions are different?

Other indices: yes, positive results on Bank Nifty and Fin Nifty, with the last-hour gain on Fin Nifty not significant (1.0%, p 0.30). Other horizons: the last-hour gain is smaller (6.2% against your specification) and an mcsGARCH given India VIX comes within 1.7% of it. Before the closing auction: the gain holds (729 test days). Other markets, other years, and tick-level data are untested. If NSE rules change again, nothing here protects against it.

## 6. What is the strongest evidence?

The hybrid beats your mcsGARCH as specified by 4.4% and 6.6% at 15 and 30 minutes (QLIKE, Diebold-Mariano p below 0.001, 792 days, registered plan P116, REAL), and rebuilt on your specification with nothing else changed, 5.1% and 7.1% (P118, five seed sets). That result survives the project-wide Benjamini-Yekutieli correction and replicates on two more indices. The second-strongest evidence is the negative one: GEX features give no out-of-sample gain under any of three sign rules, and the decomposition (P106) shows option and VIX inputs are each worth under 1% of the loss.

## 7. What would invalidate it?

- The forward test (P112, from 4 November 2026) showing the five fixes lose to the hybrid or to your specification would change the "just fix mcsGARCH" story; showing the hybrid clearly ahead would strengthen the learning story.
- An error in the evaluation chain: I found none, but timing is the likeliest place (day t-1 data for day t, confirmed by reading `features.py`).
- A flat-bar or data-vendor problem: flat days were found and excluded in one robustness check (`docs/DATA_QUALITY_FLAT_BARS.md`).
- A human referee finding a flaw in the registered-plan logic. "Registered" here means committed to git before the results were computed, with no external registry and, except for the two forward tests, on data that were not new.

## 8. Is it reproducible?

Mostly. Code, results, and a SHA-256 hash list of the raw files are committed; raw data are not, and one command (`scripts/reproduce_all.py`) downloads pinned sources and reruns the main pipeline in about 2 hours 20 minutes on one core (the full list of later studies is stated at about 13 hours, `docs/REPRODUCIBILITY.md`). Seeds are fixed and a five-seed study shows the gain is stable. I did not re-run the full pipeline in this validation; I re-derived the headline numbers from the committed forecast files instead. Another agent checks clean-clone reproduction (`reports/REPRODUCIBILITY.md`).

## 9. What is the most serious unresolved weakness?

The forecasting gain is real in-sample for 792 days but its explanation is open. Plain fixes to our mcsGARCH recover 108% to 118% of the hybrid's gain, so the evidence supports "our mcsGARCH implementation is improvable" more than "machine learning finds hidden structure". Second: the dealer sign problem is unresolved, so we cannot say whether GEX is the wrong idea or Pro positions are the wrong proxy. Third: the persistence result is NOT SETTLED (details below). Fourth: all review so far was by AI agents.

## 10. Questions for you (tailored)

**"Does your placebo test show our persistence fall is not an expiry effect?"** No, not for Nifty. Your Table 2 implies the half-life of an intraday shock fell from about 339 to 72 minutes. On the inferred window we find 359 to 72 minutes with a wide interval on the before side (197 to 1,255 minutes). After adding three plain fixes the fall still holds for Nifty on the inferred window (141 to 18 minutes, bootstrap p 0.022, not after the project-wide correction) but not for Bank Nifty or on the stated window. Against 1,223 placebo pairs of windows the fall is within the normal range (p 0.19 and 0.11), and Fin Nifty, whose expiry calendar did not change, fell as much. For Nifty this is weaker than it sounds: every placebo pair that fell as far straddles a later calendar change (88 of 128) or contains a flat-data outage day (40 of 128). Our honest summary is that the test cannot tell an expiry effect from a market-wide change; only three or four separate periods stand behind it.

**"Which 'before' window did you use?"** We do not know; that is our first question to you. The stated window (November 2023 to February 2024) has 82 trading days, not 64, and does not match. Two searches of every 64-day window put it one year earlier: starting between 14 and 23 November 2022 and ending between 10 and 21 February 2023. Please do not quote a single exact window: the searches differ by about a week. Our "after" sample reproduces your 21,239 returns exactly.

**Other questions we would ask you:** Were the connectedness shares normalised by row, and with which input, lag and horizon? (No normalised setting among 5,440 reproduces your hour-level totals; the closest misses by 5.3 points on average.) Is the "Pro" category in NSE's participant data a sensible stand-in for dealers? Would your group have intraday option-chain snapshots?

## Things not to say

Do not say the hybrid beats your model "as specified" when quoting 5.8% and 7.6% (that is against our implementation). Do not say the information-matched benchmark has "the same information"; the hybrid also uses time of day and the calendar, and the 4.8% and 5.9% edge appears only since November 2024, was found post hoc and fails the project-wide correction. Do not say the expiry event study is significant (none at 5% under an honest small-sample test). Do not call the end-of-day reversal established; it is in-sample (p = 0.016 over 78 headline tests, not over all 742). The frozen 3 November forecast was inside its 80% intervals, n = 1, not evidence.
