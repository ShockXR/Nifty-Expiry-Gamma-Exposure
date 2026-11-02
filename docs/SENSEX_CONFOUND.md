# Does BSE's Sensex expiry confound the NSE expiry results?

**Status (2 November 2026).** Every result in this note is **REAL DATA** (NSE files, third-party minute bars and third-party option prices). The only **SYNTHETIC** data are in the unit tests (`tests/test_sensex_confound.py`). Script: `experiments/sensex_confound.py`. Outputs: `results/sensex_confound/` and `results/sensex_confound_stdout.txt`.

## Short answer

* **The Sensex calendar could not be verified.** BSE's and SEBI's servers are blocked from the machine this was run on, so every row of `data/external/sensex_expiry_calendar.csv` is marked "reported". The calendar still holds up well: a simple rule (Friday from the May 2023 relaunch to 3 January 2025, Tuesday to the end of August 2025, Thursday from September 2025, moved to the previous trading day when the day is a holiday) agrees with all 71 Sensex expiry days on which third-party option prices show the expiring contract trading, and the Sensex index itself shows a large end-of-day jump on exactly those weekdays.
* **The pre-stated main test did not pass.** When Sensex moved from Friday to Tuesday in January 2025, Nifty's Tuesday-minus-Friday end-of-day value rose by +0.39 log points (95% interval -0.19 to +1.03), in the predicted direction but not significant (HAC p = 0.16, placebo p = 0.09). The test had only 15 and 16 weeks and could reliably detect only a change of about 0.78 or more.
* **The other checks point the same way.** Nifty's last 30 minutes are more volatile on Sensex expiry days in all three Sensex periods. On the five Wednesdays when Sensex expired early because Thursday was a holiday, Nifty's last 30 minutes were much more volatile than on other Wednesdays (permutation p = 0.021; Bank Nifty p = 0.002). A regression that uses every calendar change at once puts the Sensex-day effect on Nifty's last 30 minutes at +0.33 log points (about 40% more variance, p < 0.001).
* **The main NSE conclusions survive the Sensex control, and some get stronger.** The event study's primary null result does not change. Nifty's new Tuesday expiry premium after September 2025 gets larger (last 30 minutes +0.39 becomes +0.62). R2's expiry-day premia get somewhat larger too. What changes is the interpretation of weekday comparisons: Friday was not a clean control day until January 2025, and Thursday after September 2025 is a Sensex expiry day, which explains why Thursday "kept" an end-of-day premium after Nifty left it.

## 1. The question, in plain words

An **expiry day** is the day on which an options contract ends. On that day, traders who sold or bought options close or hedge their positions, and the contract is settled at an official closing value of the index. Until 3 August 2026 that value was an average over the last 30 minutes of trading, so trading in the last half hour can be unusual on expiry days.

India has two stock exchanges with index options: NSE (Nifty 50, Bank Nifty) and BSE (Sensex). The Sensex has 30 large companies, almost all of which are also in the Nifty 50. So if hedging around a Sensex expiry moves those shares near the close, Nifty would move too, even on a day when no NSE contract expires.

The earlier event study ([EVENT_STUDY.md](EVENT_STUDY.md), finding 6 and Section 5.6) found extra end-of-day volatility on weekdays when no NSE index expires: Friday in 2023 to 2024, Tuesday from January to August 2025 and Thursday from September 2025. Those are the weekdays on which Sensex weekly options are reported to have expired. If that is real, Sensex expiries are a hidden factor (a "confounder") in every study that compares NSE expiry days with other days of the week.

**How we measure "extra volatility".** For each day we compute the realized variance of the last 30 minutes (the sum of squared 1-minute returns). For a chosen day of the week we take its logarithm and subtract the average logarithm of the other days of the same week. This "week-relative" value is about 0 for an ordinary day. A value of +0.33 "log points" means about $e^{0.33} - 1 \approx 40\%$ more variance than the rest of that week. Comparing a day with its own week removes calm and stormy weeks.

## 2. What data we could and could not get

Every attempt was made on 2 November 2026 from this project's cloud machine.

| What was tried | Outcome |
| --- | --- |
| BSE servers: `api.bseindia.com`, `www.bseindia.com`, `archives.bseindia.com`, the derivatives bhavcopy folder `www.bseindia.com/download/BhavCopy/Derivative/` | **Blocked.** Every request failed with "CONNECT tunnel failed, response 403" from the network proxy (a policy block, not a missing page). |
| SEBI (`www.sebi.gov.in`), for the circular of 26 May 2025 that limits expiries to Tuesday or Thursday | **Blocked** (proxy 403; the page fetcher could not resolve the host). The circular's number, title and address were found only through a web search: SEBI/HO/MRD/MRD-TPD-1/P/CIR/2025/76, "Final Settlement Day (Expiry Day) for Equity Derivatives Contracts". |
| News and broker pages that summarise BSE circulars (Business Standard, Moneylife, IANS, The Week, Angel One, HDFC Securities, FYERS, Sahi, Ventura), and mirrors of BSE notices (regalert.today, tradingqna.com) | **Blocked** from the machine. Seen only as web search results and summaries. Used as "reported" sources. |
| Hugging Face (dataset `thetrademarkk/india-index-options-1m`, which reportedly has Sensex option bars), Kaggle, Wikipedia, archive.org | **Blocked.** |
| NSE archive (`nsearchives.nseindia.com`) | Reachable (over HTTP/1.1). NSE's files describe only NSE contracts: our bhavcopies (filtered to Nifty and Bank Nifty) and daily index files contain no BSE or Sensex rows. **No help** for the Sensex calendar, as expected. |
| GitHub repository search: "sensex options bhavcopy", "bse fo bhavcopy" | No results. |
| GitHub: "bse derivatives bhavcopy" | One downloader program (`girishg4t/bhavCopy-downloader`), no data. Not used. |
| GitHub: "sensex 1 minute", "sensex minute data" | `technovusin/sensex-historical-data` (Sensex 1-minute bars from 2023, the same provider as our Nifty bars). **Used** for the calendar check only, saved as `data/raw/sensex_1min.csv` with file hashes in `data/raw/sensex_1min_source.json`. Also `sandeepkapri/Sensex-Minute-Data` (ends March 2024), not used. |
| GitHub: "sensex expiry" | `jksarkar6263/sensex-expiries` (one-line README; its file list could not be read). `cloudharsha/nifty-options-playground-2025` (a dataset description whose Sensex table stops at "Tuesday to present", so it is out of date; not used). `vicky123411/india-options-expiry-effects`, a public research project that studies the same question with a rule-based Sensex calendar; we read its README and study guide only, which report the same three Sensex periods and that 3 January 2025 was the last Friday expiry. We did not use its code, numbers or paper. |
| GitHub code search for Sensex option files | `Akashmishra06/quantResearch`, folder `data/SENSEX/`: one file per trading day from 1 January 2024 to 30 May 2025 with 1-minute prices of the nearest Sensex call and put, whose names contain the expiry date (for example `SENSEX05JAN2472100CE` expires on 5 January 2024). The repository has no README, so where the prices came from is undocumented. **Used** as evidence for 73 expiry dates (`data/external/sensex_nearest_expiry_quantresearch.csv`). The files were read through GitHub's code search, which returns the first rows of each file; no code from any repository was run. |
| GitHub: `chintanmehta21/OptionsScraper` | A 6-row test file in BSE's bhavcopy layout dated 30 April 2026. Some values look hand-made, so it was not used. |

GitHub's file-listing interface was not enabled for these third-party repositories in this session, which is why the code search was used to list files.

## 3. The Sensex calendar we built

File: `data/external/sensex_expiry_calendar.csv`. Each row is one sourced fact with its source, address and access date. **No row is "verified"**, because no primary BSE or SEBI document could be opened. The rows are:

* **Three weekday periods (reported):** Friday from the relaunch on 15 May 2023 (first weekly expiry Friday 19 May 2023) to 3 January 2025; Tuesday from 1 January 2025 (BSE circular of 28 November 2024) to 31 August 2025; Thursday for contracts expiring from 1 September 2025 (SEBI approved Tuesday for NSE and Thursday for BSE in June 2025, under its circular of 26 May 2025).
* **The holiday rule (reported):** if the expiry day is a holiday, the contract expires on the previous trading day.
* **SEBI's circular of 26 May 2025 (reported, primary document seen only through a search engine).**
* **78 individual dates (reported):** 5 named in news or broker notices (19 May 2023, 3 January 2025, 7 January 2025, 4 September 2025, 14 January 2026) and 73 seen in the third-party option prices (January 2024 to June 2025).

The script turns the weekday periods into dates (the rule) and checks them. These rule dates are written to `results/sensex_confound/sensex_expiry_days_used.csv` with a column saying they are derived, not individually sourced; they are not in the calendar file.

**Calendar check against the option prices (REAL DATA, `calendar_check.csv`).** From January 2024 to May 2025 the rule gives 74 expiry days. 72 of them appear as expiry dates in the option names; the other 2 (13 and 20 December 2024) fall in a three-week gap with no option files at all, not a contradiction. On all 71 days where the option files include the expiry day itself, the expiring contract is trading that day, and all 71 are rule dates. All 5 individually reported dates are rule dates. One correction came from this check: NSE lists the evening Muhurat session of Friday 1 November 2024 as a trading day, but Sensex expired on Thursday 31 October 2024, so special short sessions are not treated as expiry days.

**Calendar check with the Sensex index itself (REAL DATA, `calendar_check_sensex_bars.csv`).** If the dates are right, the Sensex should show its own end-of-day jump on them. The table shows Sensex minus Nifty, week-relative log variance of the last 30 minutes, averaged over weeks without a monthly or holiday-moved NSE expiry. Bold: the reported Sensex weekday.

| period | Mon | Tue | Wed | Thu | Fri |
| --- | --- | --- | --- | --- | --- |
| before the relaunch (Jan to May 2023, 13 to 15 weeks) | -0.08 | +0.07 | -0.00 | +0.05 | -0.04 |
| Friday period (May 2023 to Jan 2025) | +0.01 | -0.17 | -0.22 | -0.16 | **+0.55** |
| of which 2023 only (no option-price dates) | -0.05 | -0.08 | -0.13 | -0.12 | **+0.37** |
| Tuesday period (Jan to Aug 2025) | -0.07 | **+0.47** | -0.15 | -0.10 | -0.18 |
| of which Jun to Aug 2025 (no option-price dates, 9 to 10 weeks) | -0.15 | **+0.96** | -0.16 | -0.28 | -0.41 |
| Thursday period (Sep 2025 to Jul 2026) | -0.14 | -0.26 | +0.08 | **+0.62** | -0.25 |

On the rule's Sensex expiry days the difference is +0.54 [95% interval +0.37, +0.71] (59 weeks), +0.47 [+0.16, +0.78] (24 weeks) and +0.59 [+0.38, +0.80] (34 weeks) in the three periods (HAC p < 0.001, 0.009 and < 0.001). In every period the Sensex-specific spike sits on the reported weekday, including the months that the option prices do not cover, and it is absent before the relaunch. The two "of which" rows and the pre-relaunch row were added after the first run, as descriptive checks without tests. This is strong evidence that the rule calendar is right, but it is market evidence, not an exchange record.

## 4. How the tests work

* **Week-relative values** are built with the event study's own functions (`experiments/expiry_event_study.py` is imported, not copied), from the same minute bars and the same NSE expiry calendar.
* **HAC p-values** (Newey-West) allow for the fact that neighbouring weeks are alike. **Block-bootstrap intervals** resample runs of 4 consecutive weeks for the same reason.
* **Placebo p-values** repeat a before/after test at many dates when nothing changed and ask how often a fake event looks as large as the real one. The event study showed that this is the better-calibrated test for before/after comparisons (Section 6.2 of EVENT_STUDY.md), but the placebo dates come from only 4 or 5 non-overlapping windows, so it is coarse.
* **The main test was written into the script's docstring before the script was first run.** The idea itself came from a table made after seeing the event study's results (Section 5.6), so even a pass would be a sharper check of a noticed pattern, not an independent discovery.

## 5. Results (REAL DATA)

### 5.1 The pre-stated main test

**The natural experiment.** BSE moved the Sensex weekly expiry from Friday to Tuesday: the last Friday expiry was 3 January 2025 and the first Tuesday expiry 7 January 2025. No NSE weekly expiry changed between November 2024 and September 2025, and Nifty kept its Thursday expiry. If Sensex expiry raises Nifty's end-of-day variance, Nifty's Tuesday should gain and its Friday should lose.

**The measure.** Nifty's week-relative last-30-minute log variance on Tuesday minus the same on Friday, 26 weeks before and 26 weeks after the week of 6 January 2025, weeks without a monthly or holiday-moved NSE expiry. The difference between Tuesday and Friday of the same week is not affected by anything that happened on Wednesday or Thursday, such as the end of Bank Nifty's Wednesday weekly in November 2024. The rule fixed in advance: the hypothesis passes if the change is positive and both the HAC p and the placebo p are below 0.05.

**Table 1. Main test (REAL DATA, `ii_change_tests.csv`, first row)**

| | value |
| --- | --- |
| Weeks before / after | 15 / 16 (8 July 2024 to 30 June 2025) |
| Mean before / after | -0.26 / +0.13 |
| Change [95% bootstrap interval] | **+0.39 [-0.19, +1.03]** |
| HAC p / Mann-Whitney p | 0.157 / 0.199 |
| Placebo p (203 placebo weeks, 5 non-overlapping windows) | 0.088 (0.077 when every event is blocked) |
| Pre-trend p | 0.92 |
| Smallest change detectable with 80% power | 0.78 |
| **Decision** | **Not supported** |

The change has the predicted sign and is not significant. For a full week, a Sensex-day effect of size d moves this measure by 2.5 times d, so the estimate corresponds to about +0.16 log points per Sensex day. With 15 and 16 weeks the test could only detect a change of about 0.78 or more; the pooled estimate in Section 5.3 (+0.33 per Sensex day) would imply a change of roughly 0.84, close to that limit. So this test is too small to settle the question either way.

**The same switch for other series (REAL DATA, 5 secondary tests).** Bank Nifty last 30 minutes +0.21 [-0.36, +0.79] (HAC p 0.48, placebo p 0.53); last hour Nifty +0.23 (0.39; 0.56) and Bank Nifty +0.07 (0.78; 0.90); whole day (5-minute realized variance) Nifty +0.31 (0.17; 0.59) and Bank Nifty +0.29 (0.34; 0.42). All positive, none significant.

### 5.2 (i) Is there a premium on Sensex expiry days that are not NSE expiry days?

Here the baseline is the other days of the week that are not NSE expiry days, so the number reads "Sensex expiry day against an ordinary day of the same week".

**Table 2. Week-relative log variance on Sensex-only expiry days (REAL DATA, `i_levels_sensex_only_days.csv`)**

| index | measure | Friday period (52 weeks) | Tuesday period (23 weeks) | Thursday period (34 weeks; Bank Nifty 23) |
| --- | --- | --- | --- | --- |
| Nifty 50 | last 30 minutes | +0.41 [+0.23, +0.60], p < 0.001 | +0.27 [+0.01, +0.56], p 0.056 | +0.63 [+0.37, +0.84], p < 0.001 |
| Nifty 50 | last hour | +0.33 [+0.11, +0.53], p 0.001 | +0.20 [+0.01, +0.39], p 0.050 | +0.51 [+0.31, +0.70], p < 0.001 |
| Bank Nifty | last 30 minutes | +0.26 [+0.10, +0.43], p < 0.001 | +0.13 [-0.16, +0.42], p 0.42 | +0.55 [+0.35, +0.76], p < 0.001 |
| Bank Nifty | last hour | +0.20 [+0.02, +0.37], p 0.026 | +0.11 [-0.11, +0.30], p 0.34 | +0.43 [+0.27, +0.60], p < 0.001 |

(HAC p-values. The Thursday period stops at the closing auction, 3 August 2026; Bank Nifty bars end 22 April 2026.) Every estimate is positive and 8 of 12 have p < 0.05. On its own this does not prove a Sensex effect, because a level mixes the Sensex effect with ordinary differences between weekdays (Fridays might be livelier near the close for other reasons). The change tests are what separate the two.

### 5.3 (ii) Does the premium move when Sensex moves?

**Table 3. Change tests (REAL DATA, `ii_change_tests.csv`, `ii_holiday_wednesdays.csv`, `ii_iii_pooled_regression.csv`)**

| test | estimate | p |
| --- | --- | --- |
| S1, Friday to Tuesday (Jan 2025), Nifty (main test) | +0.39 [-0.19, +1.03] | HAC 0.157, placebo 0.088 |
| S0, relaunch (May 2023), Nifty Friday, last 30 minutes, 18 / 12 weeks | +0.38 [-0.02, +0.74] | HAC 0.066, Mann-Whitney 0.049, placebo 0.018 |
| S0, relaunch, Bank Nifty Friday, last 30 minutes | +0.02 [-0.34, +0.40] | HAC 0.91, placebo 0.86 |
| Wednesdays when Sensex expired because Thursday was a holiday, Nifty, last 30 minutes (5 against 40 other Wednesdays) | +0.82 | permutation 0.021 |
| Same, Bank Nifty (4 against 26) | +1.14 | permutation 0.002 |
| Pooled regression, Sensex day, Nifty, last 30 minutes | +0.33 [+0.17, +0.50] | < 0.001 |
| Pooled regression, Sensex day, Nifty, last hour | +0.26 [+0.11, +0.40] | < 0.001 |
| Pooled regression, Sensex day, Bank Nifty, last 30 minutes | +0.16 [-0.01, +0.33] | 0.064 |
| Pooled regression, Sensex day, Bank Nifty, last hour | +0.13 [-0.01, +0.28] | 0.065 |

* **The relaunch (S0).** After Sensex weeklies started expiring on Fridays, Nifty's Friday end-of-day value rose by +0.38, with a placebo p of 0.018 but a HAC p of 0.066. Bank Nifty's Friday did not change. The window after the relaunch is short (12 clean weeks, stopping at Bank Nifty's move to Wednesday in September 2023).
* **Holiday-moved Wednesdays.** In the Thursday period, Sensex expired on a Wednesday five times because Thursday was a holiday (1 October 2025, 24 December 2025, 14 January 2026, 25 March 2026, 27 May 2026). Nothing about NSE's calendar marks these Wednesdays. Nifty's last 30 minutes on them averaged +0.45 against -0.37 on the other Wednesdays of that period. This test uses very few days, but it is the cleanest one: the Sensex expiry moved for a reason (a holiday) that has nothing to do with the market.
* **Pooled regression.** One regression per index over October 2022 to July 2026 (670 Nifty days in 139 weeks), with a separate level for every week, a fixed effect for each weekday, NSE expiry dummies by period and a Sensex expiry dummy, standard errors clustered by week. The Sensex dummy is identified by every change in which weekday Sensex expired on (relaunch, January 2025, September 2025, holiday moves). Nifty: +0.33 for the last 30 minutes and +0.26 for the last hour, both p < 0.001; whole day -0.03 (p 0.71). Bank Nifty: about half as large, p about 0.06. The effect sits at the end of the day, not in the whole day.

### 5.4 (iii) The NSE results with a Sensex control

The control is simple: Sensex expiry days are left out of the "rest of the week" baseline. The target day itself is never removed. All other choices are exactly those of the published analyses; the reruns reproduce the published numbers before the control is added (R2 exactly; event study to within 1e-16).

**Table 4. R2 of [REAL_DATA_RESULTS.md](REAL_DATA_RESULTS.md) (REAL DATA, `iii_r2_with_sensex_control.csv`). Expiry day against the rest of its week, Nifty; Wilcoxon p as in R2.**

| regime | measure | as published | Sensex days out of the baseline |
| --- | --- | --- | --- |
| several weeklies, Thursday (98 weeks) | whole day | +11% (p 0.045) | +15% (p 0.021) |
| one weekly per exchange, Thursday (41 / 40 weeks) | whole day | -11% (p 0.26) | -10% (p 0.36) |
| Nifty expiry on Tuesday (61 / 59 weeks) | whole day | +21% (p 0.005) | +26% (p 0.0009) |
| several weeklies, Thursday | last hour | +16% (p 0.053) | +24% (p 0.006) |
| one weekly per exchange, Thursday | last hour | +10% (p 0.65) | +17% (p 0.36) |
| Nifty expiry on Tuesday | last hour | +43% (p 0.00005) | +63% (p 0.000004) |

R2's pattern is unchanged (a premium before November 2024, none clear from November 2024 to August 2025, the largest on Tuesday), and every premium gets a little larger, because the Sensex day had been raising the baseline. The first-hour rows and the nine pairwise comparisons are in the file; the Tuesday-minus-Thursday whole-day difference goes from +0.31 (Mann-Whitney p 0.012) to +0.33 (p 0.008).

**Table 5. Key event-study tests, 6-month windows (REAL DATA, `iii_event_tests_with_sensex_control.csv`). Change [95% interval], HAC p, placebo p.**

| test | measure | as published | Sensex days out of the baselines |
| --- | --- | --- | --- |
| c, DiD main (primary test) | whole day | -0.01 [-0.61, +0.55], 0.97, 0.96 | +0.08 [-0.58, +0.70], 0.79, 0.76 |
| a, Bank Nifty Wed minus Thu | last 30 minutes | +1.14 [+0.50, +1.73], < 0.001, 0.005 | +1.19 [+0.50, +1.83], < 0.001, 0.005 |
| d, Nifty Tuesday (new expiry) | last hour | +0.49 [+0.27, +0.71], < 0.001, 0.005 | +0.67 [+0.43, +0.91], < 0.001, 0.005 |
| d, Nifty Tuesday (new expiry) | last 30 minutes | +0.39 [+0.07, +0.70], 0.026, 0.005 | +0.62 [+0.28, +0.93], < 0.001, 0.005 |
| d, Nifty Thursday (former expiry) | last 30 minutes | +0.29 [-0.05, +0.64], 0.10, 0.049 | +0.19 [-0.22, +0.59], 0.36, 0.20 |
| d, Nifty Tue minus Thu | last 30 minutes | +0.00 [-0.34, +0.39], 0.99, 0.98 | +0.30 [-0.07, +0.70], 0.16, 0.097 |
| d, Nifty Tue minus Thu | last hour | +0.44 [-0.03, +0.95], 0.10, 0.10 | +0.65 [+0.18, +1.15], 0.014, 0.054 |
| d, DiD Nifty Thu minus Bank Nifty Thu | last 30 minutes | +0.08 [-0.16, +0.30], 0.51, 0.61 | +0.01 [-0.24, +0.25], 0.93, 0.98 |

The whole-day and last-hour rows for every test are in the file. The last-hour pre-trend for Nifty's Tuesday at d is still strong (p < 0.001), as in the published table.

**Own-expiry premium by regime, Nifty, last 30 minutes** (`iii_own_premium_with_sensex_control.csv`): October 2022 to September 2023 +0.02 becomes +0.06; September 2023 to November 2024 +0.11 becomes +0.19; November 2024 to August 2025 +0.04 becomes +0.13 (none significant); September 2025 to the auction +0.42 becomes +0.59 (p < 0.001). Bank Nifty's own Wednesday premium (September 2023 to November 2024) goes from +0.32 to +0.41.

**A warning about weekday fixed effects** (`ii_iii_pooled_regression.csv`). In the pooled regression without a Sensex dummy, Nifty's own Thursday expiry appears to *lower* end-of-day variance (-0.48, p < 0.001), only because the Thursdays it is compared with (after September 2025) are Sensex expiry days. With the Sensex dummy the coefficient is -0.08 (p 0.64). Any analysis that compares weekdays across 2025 needs the Sensex calendar.

## 6. What this means for the NSE expiry results

1. **Sensex expiry is a real confounder of weekday comparisons in the last part of the day.** The main pre-stated test was too small to confirm it, but the levels, the relaunch, the holiday-moved Wednesdays, the pooled regression and the Sensex index's own spike all agree. The effect is in the last hour and the last 30 minutes, not in the whole day.
2. **The whole-day results do not depend on it.** The event study's primary test (no change in the whole-day premium when Bank Nifty's weekly ended) gives the same null result with the control, and the Sensex dummy has no whole-day effect.
3. **The NSE end-of-day results survive and mostly get stronger.** Bank Nifty's premium following its expiry from Thursday to Wednesday (event a) is unchanged. Nifty's new Tuesday premium after September 2025 is larger once Sensex days are out of the baseline, and the Tuesday-minus-Thursday switch at d goes from about zero to positive. R2's premia are a little larger. In the published versions these effects were understated, not overstated.
4. **Two puzzles in the event study are explained.** (a) Thursday "kept" an end-of-day premium after Nifty moved to Tuesday, and Bank Nifty's Thursday moved by the same amount: Sensex moved onto Thursday in the same week. With this design the Sensex effect and Nifty's departure from Thursday cannot be separated on Thursday, because both happened at once. (b) Friday, planned as a clean negative control, was a Sensex expiry day until January 2025, so its drift was partly a Sensex effect.
5. **Nifty's own Thursday expiry still shows no clear end-of-day premium from October 2022 to August 2025**, even with the control (+0.06 to +0.19, none significant). That finding of the event study stands.

## 7. Limits

* **No verified calendar.** All Sensex dates rest on reported sources: news and broker summaries of BSE and SEBI circulars, and third-party option prices whose origin is undocumented. The months without option-price coverage (May to December 2023, June 2025 onward) rely on the rule, backed only by the Sensex index's own end-of-day spike. Reading BSE's derivatives bhavcopies, which list every contract with its expiry date, would settle this; that needs network access to `www.bseindia.com`.
* **The Sensex minute bars are third-party** and could not be checked against BSE's official values. They come from the same provider as the Nifty bars, which matched NSE's official values on every checked day.
* **The main test is underpowered** (15 and 16 weeks; smallest detectable change 0.78). A longer window is not available without crossing NSE's own September 2025 change.
* **Levels mix the Sensex effect with ordinary weekday effects.** Only the change tests separate them, and they are small.
* **The holiday-moved Wednesday test uses 5 days (Bank Nifty 4).** Its p-value is small, but a single unusual day can move the mean a lot.
* **The pooled regression assumes the Sensex effect is the same from 2023 to 2026.** Sensex option trading is reported to have grown a lot after the relaunch, so the effect may have grown too.
* **Mechanism.** We show that Nifty's last minutes are more volatile on Sensex expiry days, not why. Hedging and settlement trading in the shared large stocks is the obvious candidate, but other trading that follows the Sensex calendar would look the same.
* **Overlapping weeks.** The week-relative design links the days of a week: removing a high Sensex day from the baseline raises every other day's value a little. That is the intended effect of the control, but it means the "controlled" numbers are not independent of the published ones.
* **Post hoc elements** are labelled: the hypothesis itself came from a table made after seeing results, the descriptive sub-periods of the calendar check were added after the first run, and the Muhurat-session correction was made after the first comparison with the option prices.

**Number of tests.** The script reports 83 p-values in `results/sensex_confound/all_tests.csv`: 1 main test, 3 calendar checks, 12 level tests, 13 change tests and 54 re-estimates of published NSE results with the Sensex control (18 for R2, 18 for the event study, 18 for own-expiry premia). Benjamini-Hochberg q-values are given within each family. Only the main test is confirmatory, and it did not pass. Among the 13 change tests, the holiday-moved Wednesdays (Bank Nifty q = 0.008, Nifty q = 0.069) and the pooled Nifty Sensex dummy (q = 0.001 and 0.005) remain small after the correction.

## 8. Reproduce

```bash
OMP_NUM_THREADS=1 python experiments/sensex_confound.py --fetch-sensex   # first run: downloads the Sensex bars (about 25 MB)
OMP_NUM_THREADS=1 python experiments/sensex_confound.py                  # about 1 minute, deterministic
OMP_NUM_THREADS=1 python -m pytest -q tests/test_sensex_confound.py      # unit tests on SYNTHETIC data
```

The script needs the NSE data used by the event study (`data/raw/`, see [DATA_ACQUISITION.md](DATA_ACQUISITION.md)) and `results/pipeline_nse/daily_market.csv` for the R2 rerun.

**Files.** Inputs: `data/external/sensex_expiry_calendar.csv` (sourced facts), `data/external/sensex_nearest_expiry_quantresearch.csv` (one row per trading day: the nearest Sensex contract and its expiry, with the GitHub address of each file at commit `37baff6`). Outputs in `results/sensex_confound/`: `calendar_check.csv`, `calendar_check_sensex_bars.csv`, `sensex_expiry_days_used.csv`, `i_levels_sensex_only_days.csv`, `ii_change_tests.csv`, `ii_holiday_wednesdays.csv`, `ii_iii_pooled_regression.csv`, `iii_r2_with_sensex_control.csv`, `iii_event_tests_with_sensex_control.csv`, `iii_own_premium_with_sensex_control.csv`, `all_tests.csv`, `summary.md`; the run log is `results/sensex_confound_stdout.txt`.

## Sources

Primary documents (not opened, see Section 2): SEBI circular SEBI/HO/MRD/MRD-TPD-1/P/CIR/2025/76 of 26 May 2025, https://www.sebi.gov.in/legal/circulars/may-2025/final-settlement-day-expiry-day-for-equity-derivatives-contracts_94189.html. BSE circular of 28 November 2024 (reference 20241128-70 as quoted by HDFC Securities).

Reported sources, with the address of each, are in `data/external/sensex_expiry_calendar.csv`. Related public work: vicky123411/india-options-expiry-effects on GitHub (README and STUDY_GUIDE.md read only).
