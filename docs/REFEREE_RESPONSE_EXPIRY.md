# Referee response: the expiry evidence (M8, M9, M10 and minor comments 3, 16, 18)

**Status (3 November 2026). Everything here is post hoc.** We did it in response to the referee report of 3 November 2026, after the results it questions had been seen. Every number comes from real NSE data (**REAL DATA**) unless it is marked **SYNTHETIC**. The file behind each number is named next to it.

* Script: `experiments/referee_expiry.py`.
* Outputs: `results/referee_expiry/`. The log of the run is `results/referee_expiry_stdout.txt`.
* Unit tests on synthetic data: `tests/test_referee_expiry.py`.

**Data status.**

* **NSE expiry dates are verified.** They come from NSE's full F&O bhavcopies, re-downloaded from NSE's archive for this response. Hashes are in `data/external/nse_all_index_contracts_by_expiry_source.json`, and the index rows are in `data/external/nse_all_index_contracts_by_expiry.csv.gz`.
* **BSE expiry dates (Sensex, Bankex, Sensex 50) are reported, not verified.** BSE's servers are still blocked from this machine, so these dates come from news and broker reports (`data/external/sensex_expiry_calendar.csv`, `data/external/other_weekly_expiries.csv`).

## Short answer

1. **M9 (baseline). The referee is right, and on a clean baseline most of the pre-November 2024 evidence cannot be estimated.**
   * From event a (September 2023) to event c (November 2024), a week had on average 0.10 trading days on which no index option expired on NSE or BSE. 57 of 63 weeks had none at all.
   * So no week-relative premium that excludes every expiry day can be computed in that period. Every event-study row at events a, b and c becomes inestimable.
   * The R2 "multi-weekly" premium is computable on only 17 weeks, all from January to mid-May 2023. There it is -11% for the whole day (p 0.17) and -6% for the last hour (p 0.55), against the published +11% and +16%.
   * **The Tuesday regime survives and gets larger:**
     * whole day +26% (p 0.0009);
     * last hour +63% (p below 0.001), against the published +43%.
2. **M9 (what survives at event a).** Two checks need no baseline:
   * **Raw two-day difference.** Bank Nifty Wednesday minus Bank Nifty Thursday, last 30 minutes, rose by +0.90 [+0.42, +1.36].
   * **Pooled regression with a dummy for every index's expiry.** Bank Nifty's own expiry raises its last 30 minutes by +0.61 (p below 0.001).
3. **M8 (placebo p).** Every six-month placebo p rests on at most 4 non-overlapping windows. For the end-of-day rows these all lie between March 2019 and August 2023. With 4 windows, an honest randomization p can never be below 0.20. A prediction-interval test that uses the 4 windows gives p between 0.056 and 0.080 for the four headline end-of-day rows. In 3-month windows (13 independent windows) Nifty's Tuesday last-hour rise at event d reaches the floor of 0.071 with prediction p 0.009, and event a's last-hour switch reaches the floor with prediction p 0.050.
4. **M8 (negative control).** The 6% and 9% calibration cannot be recomputed on a clean control. Once Sensex expiry Fridays are dropped, no Friday is left between event a and January 2025, and only 9 rows from event d remain. Around events a, b and c, no weekday was free of every expiry in every week of any window.
5. **M10 (noise).** On expiry days, one-minute returns in the last 30 minutes are negatively autocorrelated (Nifty about -0.08), unlike other days (about +0.03 to +0.04). This is the signature of microstructure noise, which inflates one-minute variance. With noise-robust estimators the end-of-day premia shrink:
   * The realized kernel keeps a half to three quarters of each premium. Its tests weaken to HAC p of 0.005 to 0.04 at event a and on the event-d last hour, and its placebo p rises to 0.016 to 0.11.
   * Pre-averaging gives small changes that are not significant. In simulations it was the least biased estimator, but also the noisiest.
6. **M8 (pre-trends and comparison choice).** Event a's switch has no pre-trend in 6-month windows but a strong one in 12-month windows (p 0.0007), and it survives a linear trend adjustment (+1.34, p 0.011 in 6 months). The same switch comparison at event d is null (+0.003, p 0.99, and +0.01 without a baseline), because Thursday did not lose its end-of-day premium when Nifty left it.

## Words used below

* **Expiry day:** the day an options contract ends and is settled at the official closing value of the index.
* **Week-relative premium:** the log of a day's variance minus the average log variance of the other trading days of the same week, called the **baseline**. A premium of +0.30 log points means about exp(0.30) = 1.35 times the variance of the baseline days.
* **Clean baseline:** baseline days on which no index option of NSE or BSE expires.
* **Placebo p:** we pretend a rule change happened on dates when nothing changed and count how often the fake change is as large as the real one.
* **Non-overlapping windows:** placebo dates whose before-and-after windows do not share any week. Only these are independent pieces of evidence.
* **Wild cluster bootstrap:** a resampling test that allows weeks within a block to be related. It is known to be unreliable when there are few blocks.
* **Microstructure noise:** small price movements caused by the trading process itself (bid-ask bounce, stale prices), not by news about value. It adds to variance measured from very frequent returns.
* **Subsampled RV, realized kernel, pre-averaging:** three standard ways to measure variance that are less sensitive to noise. They are described in Section 4.
* **Autocorrelation (rho1):** the correlation between one one-minute return and the next. Pure noise makes it negative. Stale prices make it positive.

## 1. M9: the week-relative baseline contains other indices' expiry days

### What we did

**A day-level calendar of every index-option expiry, 2019 to 2026.**

* **NSE (verified).** We downloaded NSE's full F&O bhavcopies (all contracts, not only Nifty and Bank Nifty) for every trading day from 1 January 2019 to 2 November 2026: 1,933 files. The one trading day with minute bars but no file in NSE's archive is 30 March 2021. No index had its weekly expiries on Tuesday in the weeks around it, so no expiry was assumed for that day.
* **What counts as an expiry.** An expiry of index X on day d means that options on X expiring on d appear and trade on d. It is monthly if a future of X that stays listed to its expiry also expires on d and is the last such future of the month. This rule matters for January to March 2025. NSE then withdrew long-dated futures on the old monthly dates and relisted them on the new ones, and both appear in the files.
* **BSE (reported, not verified).** Sensex, Bankex and Sensex 50 dates come from the reported weekday regimes, moved to the previous trading day when the expiry day is a holiday. The Sensex dates are identical to those of [SENSEX_CONFOUND.md](SENSEX_CONFOUND.md). Bankex monthly expiries on 25 November and 30 December 2024 are an unsourced assumption (last Monday of the month) and are marked as such.

**The new calendars.** The new calendars are `results/referee_expiry/all_index_expiries.csv` (one row per index and expiry day, with the expiring options' notional turnover) and `all_index_expiry_calendar.csv` (one row per trading day).

**Cross-checks (REAL DATA, `results/referee_expiry_stdout.txt`).**

* **Nifty and Bank Nifty.** From 20 October 2022 the new calendar is identical to the event study's own calendar: 211 Nifty and 133 Bank Nifty expiries, with identical monthly labels. Before that date the event study used a Thursday rule. That rule has one date with no expiry in the bhavcopies (19 October 2022) and otherwise agrees (189 dates).
* **FinNifty.** The reported Tuesday expiry of `other_weekly_expiries.csv` matches the bhavcopies on all 99 dates from 2023 to 19 November 2024.
* **Midcap Select, before August 2023: the reported calendar is wrong.** The reported row says Wednesday from January to August 2023. The bhavcopies show Midcap weekly options expiring on **Tuesday until 25 April 2023** and on Wednesday from 3 May 2023, so only 15 of the 33 reported dates are right.
* **Midcap Select, Monday period.** The reported Monday (21 August 2023 to 18 November 2024) matches 65 of 66 dates. The exception is the contract due on Monday 22 January 2024, a market holiday. The reported rule moves it to Friday 19 January, but the bhavcopies show it was not settled that day. Our files cover weekdays only.

**Then:**

* counted the no-expiry days in each week (`no_expiry_days_by_week.csv`, `no_expiry_days_by_regime.csv`);
* recomputed R2 (Table 5 of the paper) and the event-study premia with the baseline restricted to no-expiry days;
* added two estimates that need no baseline at all.

### Result 1: how many clean baseline days exist (REAL DATA, `no_expiry_days_by_regime.csv`)

Mean number of trading days per week on which no index option expires on NSE or BSE (all weeks):

| Period | Weeks | Mean no-expiry days | Weeks with 0 | Weeks with 2 or more | Usual free weekdays |
| --- | --- | --- | --- | --- | --- |
| Mar 2019 to Dec 2020 (Nifty, Bank Nifty Thursday only) | 96 | 3.70 | 0 | 96 | Mon, Tue, Wed, Fri |
| Jan 2021 to Oct 2022 (FinNifty weeklies start) | 93 | 3.17 | 0 | 91 | Mon, Wed, Fri |
| 17 Oct 2022 to 14 May 2023 | 30 | 2.60 | 0 | 27 | Mon, Wed, Fri |
| 15 May to 20 Aug 2023 (Sensex and Bankex relaunched on Friday) | 14 | 0.93 | 1 | 0 | Mon |
| 21 Aug to 3 Sep 2023 (Midcap moved to Monday) | 2 | 1.00 | 0 | 0 | Wed |
| **4 Sep 2023 to 17 Nov 2024 (event a to event c)** | **63** | **0.10** | **57** | **0** | **none** |
| 18 Nov 2024 to 5 Jan 2025 (Sensex still on Friday) | 7 | 1.57 | 2 | 3 | Mon, Tue, Wed |
| 6 Jan to 31 Aug 2025 (Sensex on Tuesday) | 34 | 2.74 | 0 | 33 | Mon, Wed, Fri |
| 1 Sep 2025 to 2 Aug 2026 (Nifty Tuesday, Sensex Thursday) | 48 | 2.69 | 0 | 46 | Mon, Wed, Fri |
| From 3 Aug 2026 (closing auction) | 14 | 2.64 | 0 | 13 | Mon, Wed, Fri |

In the event study's "clean weekly" weeks between events a and c, the count is 0 in all 42 weeks. With NSE expiries alone, the mean in that period is 0.83 days, so the BSE expiries (reported) take most of the remaining days. Over R2's whole multi-weekly regime (January 2023 to 19 November 2024, 98 weeks), the mean is 0.70. 58 weeks have no clean day, and only 17 have the two clean days that R2 requires.

**How much each expiry carried (REAL DATA, `nse_index_expiry_activity_by_segment.csv`).** This table gives the median notional turnover of the expiring options on the expiry day, relative to Nifty's median expiry in the same period.

* **Midcap Select.**
  * Before May 2023 it was negligible (about 0.00002 of Nifty).
  * On Wednesdays from May to 20 August 2023 it was 0.075.
  * On Mondays from 21 August to 3 September 2023 it was 0.21, and from event a to event c it was 0.27.
  * So the Midcap move to Monday, two weeks before event a, also turned Monday into a real expiry day. That is a genuine change inside event a's windows.
* **FinNifty** carried 0.72 to 1.11 of Nifty's expiry turnover on its Tuesday from 2022 to November 2024.
* **Nifty Next 50** never exceeded 0.005.

### Result 2: R2 (Table 5) with a clean baseline (REAL DATA, `r2_clean_baseline.csv`)

Nifty expiry day against the rest of its week. Percentages are exp(mean) minus 1. The intervals are R2's bootstrap intervals in log points, and p is the signed-rank p. "Clean" means no expiry of any index, NSE or BSE, with at least 2 baseline days, as in R2.

| Regime | Measure | Published | Weeks | Clean baseline | Weeks |
| --- | --- | --- | --- | --- | --- |
| Multi-weekly (to 19 Nov 2024) | whole day | +11.5% (p 0.045) | 98 | -10.8% [-0.24, +0.02] (p 0.17) | 17 |
| Multi-weekly | last hour | +16.4% (p 0.053) | 98 | -6.3% [-0.24, +0.12] (p 0.55) | 17 |
| Multi-weekly, 15 May 2023 to 19 Nov 2024 | both | (part of the row above) | | not estimable (0 weeks) | 0 |
| Thursday single-weekly | whole day | -11.2% (p 0.26) | 41 | -13.1% [-0.36, +0.08] (p 0.27) | 36 |
| Thursday single-weekly | last hour | +9.6% (p 0.65) | 41 | +24.4% [-0.13, +0.56] (p 0.20) | 36 |
| Tuesday (from 1 Sep 2025) | whole day | +21.1% (p 0.005) | 61 | +26.0% [+0.12, +0.36] (p 0.0009) | 59 |
| Tuesday | last hour | +42.8% (p below 0.001) | 61 | +62.6% [+0.31, +0.65] (p below 0.001) | 59 |
| Tuesday, before 3 Aug 2026 | whole day | +23.5% (p 0.005) | 48 | +28.8% (p 0.0009) | 46 |
| Tuesday, before 3 Aug 2026 | last hour | +42.2% (p below 0.001) | 48 | +65.8% (p below 0.001) | 46 |
| Tuesday, closing-auction days | last hour | +45.2% (p 0.19) | 13 | +51.7% (p 0.11) | 13 |

All 17 clean multi-weekly weeks fall between 2 January and 14 May 2023, when only Nifty, Bank Nifty, FinNifty, an inactive Midcap and (reported) Sensex 50 on Nifty's Thursday expired. Two intermediate versions are in the same file:

* With only NSE expiries removed, the multi-weekly whole-day premium is +9.3% (36 weeks, p 0.22).
* With only one clean baseline day required, it is +11.0% (40 weeks, p 0.19), and the last hour is +19.4% (p 0.053).

**Minor comment 16: Tuesday against Thursday single-weekly** (REAL DATA, `r2_tue_vs_thu_by_baseline.csv`).

* **Whole day.** The two regimes differ under every baseline. The Mann-Whitney p is 0.012 (published baseline), 0.008 (Sensex days removed), 0.007 (NSE expiries removed) and 0.004 (clean baseline).
* **After correction.** Multiplied by the 18 tests in the family (Bonferroni), these become 0.21, 0.14, 0.12 and 0.08. None survives a family-wise correction.
* **Last hour.** The difference is not significant under any baseline (p 0.07 to 0.21).

### Result 3: the event study with a clean baseline (REAL DATA, `event_tests_clean_baseline.csv`)

The tests require at least 3 clean-baseline weeks on each side.

* **Events a, b and c.** None of the rows can be estimated in any window (3, 6 or 12 months). For event a the after side has 0 usable weeks, and for c the before side has 0. This includes the pre-specified primary test (event c).
* **Event d.** All 40 rows can be estimated. The 6-month results:

| Event d row (6 months) | Published | Clean baseline |
| --- | --- | --- |
| Nifty Tuesday, last hour | +0.49 (placebo p 0.005, pre-trend p 1.1e-5) | +0.52 [+0.26, +0.77], HAC p 0.0002, placebo p 0.005, pre-trend p 2e-9 |
| Nifty Tuesday, last 30 minutes | +0.39 (placebo p 0.005) | +0.53 [+0.11, +0.90], HAC p 0.014, placebo p 0.005 |
| Nifty Thursday, last 30 minutes | +0.29 (placebo p 0.049) | +0.38 [-0.05, +0.79], HAC p 0.076, placebo p 0.027 |
| Nifty Tuesday minus Thursday (switch), last 30 minutes | +0.003 (p 0.99) | +0.02 [-0.26, +0.34], HAC p 0.91 |
| Nifty Tuesday, last hour, 12 months | +0.20 (HAC p 0.13) | +0.25 [-0.02, +0.50], HAC p 0.073 |

**Two estimates that need no baseline** (REAL DATA).

* **Raw two-day difference** (`two_day_raw_differences.csv`). For a switch comparison (new expiry weekday minus old one, same index), in a full five-day week the week-relative switch equals 1.25 times the raw difference of the two days' log variances: the three other days enter both baselines equally and cancel. So the switch hardly depends on which baseline days are used.
  * **Event a, Bank Nifty Wednesday minus Thursday** (19 and 20 weeks):
    * last 30 minutes rose by +0.90 [+0.42, +1.36], HAC p 0.0001;
    * last hour rose by +0.72 [+0.34, +1.08];
    * whole day (rv5) +0.09 [-0.18, +0.37].
  * **The same difference for Nifty, which did not change its expiry at event a,** did not move: last 30 minutes +0.05 (HAC p 0.84).
  * **At event d, Nifty Tuesday minus Thursday:** last 30 minutes +0.01 [-0.26, +0.30], last hour +0.35 [-0.03, +0.77].
* **Pooled regression with every index's expiry as a dummy** (`pooled_all_index_regression.csv`). This keeps all days and controls for each index's expiry with its own dummy. It uses week fixed effects, weekday effects, standard errors clustered by week, and clean weekly weeks from 17 October 2022 to the closing auction.
  * **Bank Nifty, last 30 minutes** (618 days, 129 weeks):
    * own Wednesday expiry (event a to c) +0.61 (se 0.14, p below 0.001);
    * own Thursday expiry (before a) +0.42 (p 0.010);
    * Sensex expiry days +0.26 (p 0.010).
  * **Nifty, last 30 minutes** (670 days, 139 weeks):
    * own Tuesday expiry (from d) +0.52 (p 0.006);
    * Sensex expiry days +0.30 (p 0.008);
    * Bank Nifty expiry days +0.30 (p 0.011).
  * **Nifty's Thursday expiry before d** is not separable from a Thursday weekday effect once Sensex expires on Thursday after d: -0.11 (p 0.57), or +0.19 (p 0.018) without weekday effects.

### Result 4: the calendar changes the paper omitted (REAL DATA for NSE rows, `calendar_changes.csv`)

Every change from 2023 to 2026 that touches the weekdays compared in the paper. "Inside windows" lists the event windows (6 or 12 months each side, bounded by the neighbouring events as in the event study) that contain the change.

| Date | Exchange | Index | Change | Status | Inside windows |
| --- | --- | --- | --- | --- | --- |
| 3 May 2023 | NSE | Midcap Select | weekly expiry Tuesday to Wednesday | verified | a 6m, a 12m |
| 15 May 2023 | BSE | Sensex, Bankex | weekly options relaunched, Friday expiry (first on 19 May 2023) | reported | a 6m, a 12m |
| 21 Aug 2023 | NSE | Midcap Select | weekly expiry Wednesday to Monday (two weeks before event a) | verified | a 6m, a 12m |
| 28 Aug 2023 | NSE | Midcap Select | monthly expiry to Monday | verified | a 6m, a 12m |
| 6 Sep 2023 | NSE | Bank Nifty | weekly Thursday to Wednesday (**event a**) | verified | event a's own week, b 6m, b 12m |
| 16 Oct 2023 | BSE | Bankex | weekly expiry Friday to Monday | reported | a 6m, a 12m, b 6m, b 12m |
| 27 Mar 2024 | NSE | Bank Nifty | monthly Thursday to Wednesday (**event b**) | verified | a 12m, b 6m, b 12m, c 12m |
| 24 Apr 2024 | NSE | Nifty Next 50 | index options listed (monthly) | verified | a 12m, b 6m, b 12m, c 12m |
| 13 Nov 2024 | NSE | Bank Nifty | last weekly expiry (**event c**) | verified | b 12m, c 6m, c 12m |
| 14 Nov 2024 | BSE | Sensex 50 | last weekly expiry | reported | b 12m, c 6m, c 12m |
| 18 and 19 Nov 2024 | NSE, BSE | Midcap Select, Bankex, FinNifty | last weekly expiries | verified (NSE), reported (BSE) | event c's own week, d 12m |
| 20 Nov 2024 | SEBI | all index options | phase 1: one weekly expiry per exchange, minimum contract value Rs 15 lakh, extra 2% margin on short options on expiry day | verified (circular) | event c's own week, d 12m |
| 7 Jan 2025 | BSE | Sensex | weekly expiry Friday to Tuesday (last Friday 3 Jan 2025) | reported | c 6m, c 12m, d 12m |
| 30 Jan 2025 | NSE | Bank Nifty, FinNifty, Midcap Select, Nifty Next 50 | monthlies to last Thursday (**c2** for Bank Nifty) | verified | c 6m, c 12m, d 12m |
| 1 Feb 2025 | SEBI | all index options | phase 2: upfront collection of option premium from buyers; no calendar-spread margin benefit on expiry day | reported (secondary summary) | c 6m, c 12m, d 12m |
| 1 Apr 2025 | SEBI | all index options | phase 3: intraday monitoring of position limits | reported (secondary summary) | c 6m, c 12m, d 6m, d 12m |
| 26 May 2025 | SEBI | all equity derivatives | circular: each exchange's expiries on Tuesday or Thursday | reported | c 12m, d 6m, d 12m |
| 2 Sep 2025 | NSE | Nifty 50 | weekly Thursday to Tuesday (**event d**) | verified | event d's own week |
| 4 Sep 2025 | BSE | Sensex | weekly expiry Tuesday to Thursday | reported | event d's own week |
| 30 Sep 2025 | NSE | Nifty, Bank Nifty, FinNifty, Midcap Select, Nifty Next 50 | monthlies to Tuesday | verified | d 6m, d 12m |
| 3 Aug 2026 | NSE | cash market | closing auction (**event e**) | detected in minute bars | |
| 12 Aug 2026 | NSE | NIFTYFPI | index options listed (monthly) | verified | |

**Sources.**

* NSE rows: the bhavcopy detection, in which three expiries on a new weekday confirm a change (`nse_schedule_changes_detected.csv`).
* SEBI phase 1: circular SEBI/HO/MRD/TPD-1/P/CIR/2024/132 of 1 October 2024, listed as verified in `docs/LITERATURE_REVIEW.md`.
* SEBI phases 2 and 3: a secondary summary (Business Standard, 3 October 2024). The circular itself was not re-read for this response.
* BSE rows: the source and address of each are in `calendar_changes.csv`.

### What it changes

* **Table 5 and Section 6.3.1.** The multi-weekly premium (+11%, p 0.045) has to be described as baseline-dependent. On a clean baseline it can be computed only for January to mid-May 2023, where it is negative and not significant. For the 18 months that make up most of that regime it cannot be computed at all.
  * The regime comparisons that involve the multi-weekly regime cannot be interpreted as expiry effects.
  * The Tuesday result is robust and larger on a clean baseline.
  * The Tuesday-Thursday regime difference (minor 16) is nominally significant under every baseline but survives no family-wise correction.
* **Table 6 and "the clearest causal evidence".** Event a cannot be tested on a clean week-relative baseline. The evidence for event a now rests on two estimates that do not use a baseline: the raw two-day difference (+0.90 in the last 30 minutes) and the pooled regression (+0.61). Both give the same answer as the published test. The text should say that it rests on these.
  * Midcap Select's move to Monday (21 August 2023), which made Monday a real expiry day, and Bankex's move to Monday (16 October 2023) fall inside event a's windows. They affect the week-relative values through Monday, which the switch comparison cancels.
* **Table 1 and Section 3** should carry the table above. "Event c is therefore a bundle of changes" should list the four other weekly series that stopped in the same week (FinNifty, Midcap Select, Bankex, and Sensex 50 a week earlier) and SEBI's 2025 phases.

## 2. M8: placebo p-values with few effective units

### What we did

**Recomputed every row of `results/event_study/event_tests.csv`** (1,308 rows). The change, the placebo p and the number of non-overlapping placebo windows (`ri_n_independent`) are identical to the published ones. Next to the placebo p, every row now has the following (`placebo_inference_all_rows.csv`):

* **`ri_n_independent`** and the first and last placebo week.
* **A non-overlapping randomization p** (`nonoverlap_p`). It uses only placebo windows that share no week with each other: the q windows that the published count finds, with centres listed. Its smallest possible value, `nonoverlap_floor` = 1/(q + 1), is shown next to it.
* **A prediction-interval p** (`prediction_t_p`). It treats the q placebo t statistics as draws from a normal distribution and asks how unusual the real t is as one more draw: (t - mean) / (sd times the square root of 1 + 1/q), against a t distribution with q - 1 degrees of freedom. This is in the spirit of Conley and Taber (2011): it uses the comparison units to learn the null distribution. It needs q of at least 3 and is valid only if the placebo t statistics are roughly normal.
* **A wild cluster bootstrap p** (`wild_cluster_p`, with the number of clusters G). It is the restricted version with Webb's six-point weights. Its clusters are blocks of 4 consecutive weeks within each side of the event.
  * **Caveat:** with G of about 10 this is the few-cluster case.
  * It protects against correlation within a block. It does not protect against slow drifts in the weekday premium, which is the problem the negative control revealed.
* **A pre-trend p and a trend-adjusted change.** The trend-adjusted change is the level shift at the event after one linear trend is fitted through both sides, with a HAC p.

### Result (REAL DATA, `placebo_inference_summary.csv`, `placebo_inference_headline_rows.csv`)

| Window | Rows | Non-overlapping windows (median, max) | Floor of the honest randomization p | Share with placebo p below 0.05 | Non-overlap p below 0.05 | Prediction p below 0.05 | Wild cluster p below 0.05 (median G) | HAC p below 0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 months | 404 | 13, 13 | 0.071 | 22.2% | 0% | 16.9% | 23.9% (6) | 39.6% |
| 6 months | 404 | 4, 4 | 0.20 | 13.2% | 0% | 10.3% | 23.3% (10) | 37.4% |
| 12 months | 404 | 2, 2 | 0.33 | not computed (fewer than 3 windows) | 0% | not computed | 26.2% (16) | 34.9% |

**The placebo windows.** For the minute-bar rows of Table 6, all placebo windows lie between 4 March 2019 and 28 August 2023. In 6-month windows the 4 non-overlapping windows are centred on 2 September 2019, 31 August 2020, 30 August 2021 and 29 August 2022. So the placebo distribution describes the quiet pre-2023 calendar, not the period of the events.

**Headline rows, 6 months:**

| Row | Change | Published placebo p | Windows q | Non-overlap p (floor) | Prediction p | Wild cluster p (G) |
| --- | --- | --- | --- | --- | --- | --- |
| a: Bank Nifty switch, last 30 minutes | +1.14 | 0.005 | 4 | 0.20 (0.20) | 0.080 | 0.005 (10) |
| a: Bank Nifty switch, last hour | +0.92 | 0.005 | 4 | 0.20 (0.20) | 0.066 | 0.006 (10) |
| d: Nifty Tuesday, last hour | +0.49 | 0.005 | 4 | 0.20 (0.20) | 0.056 | 0.005 (10) |
| d: Nifty Tuesday, last 30 minutes | +0.39 | 0.005 | 4 | 0.20 (0.20) | 0.066 | 0.009 (10) |
| d: Nifty Thursday, last 30 minutes | +0.29 | 0.049 | 4 | 0.20 (0.20) | 0.011 | 0.17 (10) |
| c: primary DiD, whole day | -0.01 | 0.96 | 4 | 1.00 (0.20) | 0.92 | 0.97 (9) |
| a: Nifty Friday, whole day (negative control) | -0.41 | 0.005 | 4 | 0.20 (0.20) | 0.008 | 0.014 (10) |

In 12-month windows only 2 non-overlapping windows exist (floor 0.33). Event a's switch in the last 30 minutes (+0.99) then has wild cluster p 0.004 with G = 19.

**3-month windows** have 13 non-overlapping windows (floor 0.071), so the honest tests have more room. The same rows:

| Row (3 months) | Change | Non-overlap p (floor 0.071) | Prediction p | Wild cluster p (G) |
| --- | --- | --- | --- | --- |
| a: Bank Nifty switch, last 30 minutes | +1.44 | 0.14 | 0.11 | 0.024 (6) |
| a: Bank Nifty switch, last hour | +1.13 | 0.071 | 0.050 | 0.015 (6) |
| d: Nifty Tuesday, last hour | +0.46 | 0.071 | 0.009 | 0.044 (6) |
| d: Nifty Tuesday, last 30 minutes | +0.45 | 0.14 | 0.13 | 0.079 (6) |
| c: primary DiD, whole day | +0.12 | 0.86 | 0.83 | 0.82 (6) |
| a: Nifty Friday, whole day (negative control) | -0.45 | 0.21 | 0.13 | 0.010 (6) |

In 3-month windows the prediction test rejected none of the 35 published Friday rows (Section 3).

### What it changes

* **"A placebo p of 0.005" is not a 0.5% tail.** It is the floor of a distribution with 4 effective draws, and the honest randomization p of every 6-month row is at least 0.20.
* **The prediction-interval test is the most usable small-sample alternative.** It gives p between 0.056 and 0.080 for the four rows the paper calls solid. That is suggestive, not decisive.
* **The wild cluster bootstrap would call them significant, but it does not pass the negative control.** It rejects in 17% of the published 6-month Friday rows (Section 3), so it over-rejects about threefold here.
* **The rule "solid only with small placebo p" should be restated.** In 6-month windows none of the four end-of-day results that the paper calls solid is significant at 5% under these tests: the honest randomization p is at least 0.20 and the prediction p is 0.056 to 0.080. The prediction test is not conservative either: it rejects in 10.3% of all 6-month rows and in 14.3% of the 6-month Friday rows.
* **The best-supported single result is Nifty's Tuesday last hour at event d in 3-month windows.** It sits at the floor of the honest randomization p (0.071, the smallest possible with 13 windows), and its prediction p is 0.009, in the window where the prediction test rejected no Friday row. Event a's last-hour switch is borderline there (floor 0.071, prediction p 0.050). The 3-month windows are short (9 or 10 weeks a side), and that row's 3-month pre-trend p is 0.045 (`placebo_inference_all_rows.csv`).
* **Minor 18.** The Table 6 note should read: "the placebo p compares the change with about 184 dates without a rule change, drawn from 4 non-overlapping windows in March 2019 to August 2023 (floor about 0.005; the smallest p attainable with 4 independent windows is 0.20)".

## 3. M8: the negative control, recomputed without Sensex Fridays

### What we did

We reran the published Friday rows in three versions. These are `a_N_Fri`, `a_BN_Fri`, `c_DiD_Fri` and `d_N_Fri`, with the event study's 9 measures that do not use option or futures turnover (8 for the DiD), making 35 rows per window. The versions are:

* with Fridays on which Sensex expired (reported) dropped;
* with Fridays on which any index expired dropped;
* the same, with the clean baseline as well.

We also checked, for every event and window, whether any weekday was free of every NSE and BSE expiry in every week (`weekday_free_of_expiry.csv`, `weekday_free_summary.csv`).

### Result (REAL DATA, `negative_control_recomputed.csv`, `negative_control_rows.csv`, `weekday_free_summary.csv`)

**Published Friday rows (35 per window), reproduced.**

* The published shares come out exactly: HAC p below 0.05 in 20.0%, 34.3% and 45.7% (3, 6, 12 months), and placebo p below 0.05 in 5.7% and 8.6%.
* The new tests on the same rows reject as follows:

| New test | 3 months | 6 months | 12 months |
| --- | --- | --- | --- |
| Wild cluster bootstrap | 14.3% | 17.1% | 25.7% |
| Prediction-interval test | 0% | 14.3% | not computed |
| Non-overlap test | 0% | 0% | 0% |

**Without Sensex expiry Fridays.**

* **Only event d's rows remain.** Only 9 of the 35 rows per window can be estimated, all of them event d's Nifty Friday. From event a to 3 January 2025 every Friday was a Sensex expiry day (reported), so event a's after side and event c's before side have no Friday left.
* **In these 9 rows:**
  * the placebo p is never below 0.05 (0 of 9 in 3 and 6 months);
  * HAC p is below 0.05 in 0 of 9, 3 of 9 and 5 of 9 (3, 6, 12 months);
  * the wild cluster p is below 0.05 in 0, 0 and 4 of 9.
* **With the clean baseline as well,** HAC p is below 0.05 in 0 of 9, 1 of 9 and 4 of 9, and the placebo p in 1 of 9 (3 months) and 0 of 9 (6 months).

**Was any weekday free of every expiry?** For events a, b and c: **no weekday, on either side, in any window** (3, 6 or 12 months).

* For example, around event a (6 months) Monday was free in 20 of 25 weeks before and 0 of 22 after, and Friday in 8 of 24 before and 0 of 25 after.
* Only event d has free weekdays. Before it, Monday, Wednesday and Friday were free in all weeks of the 3-month window, and Monday and Friday in the 6-month window. After it, Friday was free in every week of every window.

### What it changes

* **The 6% and 9% placebo rejection rates cannot be confirmed on a clean negative control.** The published Friday rows at events a and c compare Sensex expiry Fridays with Fridays before the Sensex relaunch. They measure Sensex's arrival, not the test's false-positive rate: Nifty's Friday whole-day change of -0.41 at event a is one of them. What is left (9 rows from one event) is too few for a calibration.
* **The referee's point stands as stated.** Around events a, b and c no weekday was free of every NSE and BSE expiry, so no clean weekday control existed. The paper should say so in Sections 5.4 and 6.3.2 and drop "Friday, which no NSE rule targeted, is a negative control" as a calibration device.

## 4. M10: noise-robust end-of-day variance

### What we did

For every Nifty and Bank Nifty day we computed the variance of the last 30 minutes (15:00 to 15:30, 30 one-minute returns) and of the last hour (14:30 to 15:30, 60 returns) in five ways (`noise_robust_daily.csv.gz`):

* **rv1:** one-minute realized variance, as in the paper.
* **rvss2, rvss5:** realized variance of 2-minute and 5-minute returns, computed from every possible starting minute and averaged ("subsampled").
* **rk:** a realized kernel (Barndorff-Nielsen, Hansen, Lunde and Shephard 2008) with Parzen weights and 3 lags of one-minute returns. Versions with 1 and 5 lags are in the daily file.
* **pav:** a pre-averaging estimator (Jacod, Li, Mykland, Podolskij and Vetter 2009), self-implemented. It averages 6 (last 30 minutes) or 8 (last hour) consecutive one-minute returns with a tent-shaped weight, subtracts the noise bias, and corrects for the small-window loss. It can be negative on a single day; that happened on 26 of 2,358 Nifty days and 12 of 2,283 Bank Nifty days in the last 30 minutes, and those days drop out of the log tables.

We also computed the first-order autocorrelation of one-minute returns (rho1) in the last 30 minutes, the last hour and a midday reference window (12:00 to 12:30), by type of day (`autocorrelation_1min_returns.csv`).

**Days from 3 August 2026** (closing auction) are kept apart in every table. On these days the index stands still from 15:16 to 15:28. The last 30 minutes of Nifty contain on average 13 zero one-minute returns (`autocorrelation_1min_returns.csv`), so variance in that window measures mostly the final jump and is not comparable with earlier days.

**Checks.**

* The one-minute versions reproduce the event study's `rv_last30` and `rv_last_hour` and the pipeline's `rv_last_hour` to within 1e-11 (relative), as reported in the stdout.
* Before using the estimators on real data we tested them on simulated prices (SYNTHETIC, `noise_estimator_simulation.csv`). There are 400 simulated 30-minute windows, built from a one-second efficient price.

**Simulation results (SYNTHETIC):** mean estimate divided by the true variance.

| Scenario (SYNTHETIC) | rho1 of 1-minute returns | rv1 | rvss2 | rvss5 | rk | pav |
| --- | --- | --- | --- | --- | --- | --- |
| no noise | +0.008 | 1.009 | 1.018 | 1.010 | 1.021 | 1.015 |
| independent noise, small | -0.034 | 1.114 | 1.072 | 1.038 | 1.052 | 1.019 |
| independent noise, large | -0.148 | 1.482 | 1.242 | 1.102 | 1.142 | 1.005 |
| stale index (20 stocks, each updating every 30 seconds on average) | +0.250 | 0.593 | 0.757 | 0.906 | 0.839 | 0.971 |

**What the simulation shows.**

* **Pre-averaging is the least biased in every scenario,** but it is the noisiest. With no noise, its day-to-day spread is 0.526 times the true value, against 0.265 for rv1 and 0.367 for the kernel.
* **The sign of rho1 tells which kind of noise is present.** It is negative for bid-ask-type noise, which inflates rv1, and positive for stale prices, which deflate it.

### Result 1: autocorrelation (REAL DATA, `autocorrelation_1min_returns.csv`)

Nifty, last 30 minutes, mean rho1 per day:

| Period | Own expiry day | Days with no expiry of any index | Difference (Welch p) |
| --- | --- | --- | --- |
| Oct 2022 to a | +0.039 (46 days) | +0.033 (93) | +0.007 (0.88) |
| a to c | -0.092 (63) | +0.095 (6) | -0.19 (0.085) |
| c to d | -0.066 (41) | +0.031 (104) | -0.097 (0.009) |
| d to e (Tuesday expiry) | -0.082 (48) | +0.040 (129) | -0.122 (0.001) |
| from e (closing auction) | -0.015 (13) | +0.012 (37) | -0.027 (0.52) |

Days with only another index's expiry lie in between or close to the expiry days: -0.039 from a to c (225 days) and -0.076 after d (48 days, mostly Sensex Thursdays).

* At midday there is no such difference. After d, the own expiry day has +0.026 and no-expiry days +0.055 (p 0.35).
* Bank Nifty from a to c: own expiry -0.142 (63 days) against -0.081 on its other days (p 0.061).
* So since September 2023 the last 30 minutes of an expiry day show noise of the bid-ask type that ordinary days do not.

### Result 2: R2's Tuesday regime with noise-robust estimators (REAL DATA, `noise_robust_r2.csv`)

Nifty Tuesday expiry against the rest of its week, 1 September 2025 to 2 August 2026 (before the auction). Mean log difference, with the signed-rank p in brackets:

| Window, baseline | rv1 | rvss2 | rvss5 | rk | pav |
| --- | --- | --- | --- | --- | --- |
| last hour, published baseline (48 weeks) | +0.35 (below 0.001) | +0.31 (0.0004) | +0.20 (0.062) | +0.25 (0.006) | +0.13 (0.24) |
| last 30 minutes, published baseline (48 weeks; pav 46) | +0.42 (0.0003) | +0.33 (0.005) | +0.16 (0.16) | +0.22 (0.082) | +0.03 (0.85) |
| last hour, clean baseline (46 weeks) | +0.51 (below 0.001) | +0.43 (below 0.001) | +0.30 (0.007) | +0.35 (0.0009) | +0.21 (0.084) |
| last 30 minutes, clean baseline (46 weeks; pav 44) | +0.59 (below 0.001) | +0.49 (0.0007) | +0.26 (0.057) | +0.33 (0.023) | +0.13 (0.35) |

On the 13 closing-auction weeks, the last-hour values are +0.37 (rv1, p 0.19) and +0.35 (rk, p 0.24).

### Result 3: event tests with noise-robust estimators (REAL DATA, `noise_robust_event_tests.csv`)

Six-month windows, change [95% interval], with HAC p and placebo p:

| Row | rv1 (published change; interval recomputed) | rvss5 | rk | pav |
| --- | --- | --- | --- | --- |
| a: Bank Nifty switch, last 30 minutes | +1.14 [+0.52, +1.74]; 0.0001, 0.005 | +0.87 [+0.09, +1.61]; 0.022, 0.10 | +0.78 [+0.02, +1.49]; 0.037, 0.087 | +0.38 [-1.03, +1.62]; 0.55, 0.65 |
| a: Bank Nifty switch, last hour | +0.92 [+0.41, +1.39]; 0.0001, 0.005 | +0.55 [-0.06, +1.11]; 0.051, 0.28 | +0.62 [+0.15, +1.07]; 0.008, 0.11 | +0.37 [-0.32, +1.05]; 0.28, 0.51 |
| d: Nifty Tuesday, last hour | +0.49 [+0.27, +0.71]; 0.0001, 0.005 | +0.29 [+0.03, +0.55]; 0.034, 0.070 | +0.37 [+0.12, +0.62]; 0.005, 0.016 | +0.17 [-0.13, +0.50]; 0.29, 0.49 |
| d: Nifty Tuesday, last 30 minutes | +0.39 [+0.06, +0.69]; 0.026, 0.005 | +0.04 [-0.35, +0.44]; 0.84, 0.87 | +0.21 [-0.15, +0.61]; 0.27, 0.24 | -0.17 [-0.69, +0.39]; 0.48, 0.60 |

**Direct check of the noise share** (REAL DATA, `noise_robust_premium_by_regime.csv`). The week-relative premium of the log ratio rv1 / rk measures how much more one-minute variance exceeds the kernel on the expiry day than on the rest of the week. It is:

* +0.27 [+0.12, +0.42] for Nifty after d;
* +0.25 [+0.11, +0.40] for Nifty from a to c;
* +0.19 [+0.04, +0.32] for Bank Nifty from a to c.

The same table shows Nifty's own-expiry last-30-minute premium after d: +0.42 with rv1, +0.15 [-0.06, +0.39] with the kernel, +0.07 with rvss5 and -0.03 with pre-averaging. In contrast, Bank Nifty's Thursday expiry premium before event a (October 2022 to August 2023) holds under every estimator: rv1 +0.32, rvss5 +0.26, rk +0.21, pav +0.27, each with an interval above zero.

### What it changes

* **The referee's concern is borne out.** Part of the end-of-day premium measured with one-minute returns is noise of the bid-ask type that appears on expiry days. The negative autocorrelation and the rv1-over-kernel ratio show it directly.
* **"Expiry raises volatility at the end of the day" must be qualified.** With the realized kernel, which removes most of the noise bias in the simulation (+14% with large noise, against +48% for rv1), the event-a switch keeps about two thirds of its size: +0.78 in the last 30 minutes and +0.62 in the last hour. Its placebo p rises to 0.087 and 0.11.
* **The event-d Tuesday last hour keeps three quarters of its size:** +0.37, placebo p 0.016.
* **The event-d Tuesday last 30 minutes loses most of it:** +0.21, p 0.27.
* **With pre-averaging, the least biased but noisiest estimator, no change is significant.**
* **The robust statement is narrower.** On expiry days the last half hour shows more price movement, and more of it reverses within a minute. Whether the efficient-price variance rises is uncertain with 30 to 60 one-minute returns a day. For settlement risk the transient part still matters, because the settlement price averages the last 30 minutes.
* **One-minute measures should carry the same caveat as the whole-day measures in Section 6.3.3.**

## 5. M8 bullets: pre-trends and the choice of comparison

### What we did

`pretrends_and_comparisons.csv` lists, for each event row, three things: the comparison type, the window, and the pre-trend p. A comparison is a single weekday, a switch (new expiry weekday minus old, same index), a difference-in-differences between two indices, or the Friday negative control. The file also gives the trend-adjusted change.

`switch_comparison_a_vs_d.csv` puts event a's headline comparison next to its analogue at event d in five versions:

* published;
* Sensex days out of the baselines;
* clean baseline;
* raw two-day difference;
* realized kernel.

### Result (REAL DATA)

| Row | Comparison type | 6 months: change, pre-trend p | 12 months: change, pre-trend p | Trend-adjusted change (6 months; 12 months) |
| --- | --- | --- | --- | --- |
| a: Bank Nifty Wed minus Thu, last 30 minutes | switch | +1.14, 0.53 | +0.99, 0.0007 | +1.34 (p 0.011); +1.08 (p 0.088) |
| a: same, last hour | switch | +0.92, 0.58 | +0.69, 0.0011 | +0.96 (p 0.018); +0.88 (p 0.053) |
| d: Nifty Tuesday, last hour | single weekday | +0.49, 1.1e-5 | +0.20, 0.83 | +0.47 (p 0.089); +0.68 (p 0.0010) |
| d: Nifty Tuesday, last 30 minutes | single weekday | +0.39, 0.93 | +0.17, 0.91 | +0.52 (p 0.16); +0.64 (p 0.039) |
| d: Nifty Tue minus Thu, last 30 minutes | switch | +0.003, 0.28 | -0.29, 0.10 | +0.59 (p 0.14); +0.37 (p 0.23) |
| c: primary DiD, whole day | difference-in-differences | -0.01, 0.37 | -0.13, 0.84 | +0.16 (p 0.82); +0.22 (p 0.69) |

**Event d's analogue of event a's headline** (Tuesday minus Thursday, last 30 minutes, 6 months):

| Version | Change | p |
| --- | --- | --- |
| published | +0.003 | 0.99 |
| Sensex days out of the baselines | +0.30 | 0.16 (placebo 0.097) |
| clean baseline | +0.02 | 0.91 |
| raw two-day difference | +0.01 | 0.95 |
| realized kernel | -0.27 | 0.073 |

In the last hour the same comparison is +0.44 (p 0.099) published, +0.35 (p 0.10) with a clean baseline or with no baseline, and +0.22 (p 0.46) with the kernel.

### What it changes

* **Event a.** "With placebo p at its floor and no pre-trend" holds for 6-month windows only, and the paper should say so. In 12-month windows the switch has a pre-trend (p 0.0007), although a linear trend adjustment leaves +1.08 (p 0.088).
* **Event d, Tuesday last hour.** This is the row with a 6-month pre-trend p of 1.1e-5. It should not be called solid. Its trend-adjusted change is +0.47 with p 0.089.
* **One comparison type for all events.** The switch is the comparison that is least sensitive to the baseline. On it, event a is large and event d is null in the last 30 minutes.
  * The likely reason is reported, not verified: Sensex moved onto Thursday in the same week as event d, so Thursday did not lose its end-of-day premium.
  * The paper should report the switch at both events and say that it replicates at a and not at d.

## 6. Minor comments 3, 16 and 18

* **Minor 3.** The five Nifty and Bank Nifty changes are:
  * 6 September 2023 (a);
  * 27 March 2024 (b);
  * 13 November 2024 (c);
  * 30 January 2025 (c2);
  * 2 September 2025 (d).

  The other changes are in Section 1, Result 4: FinNifty, Midcap Select, Bankex, Sensex and Sensex 50, the Nifty Next 50 listing and the SEBI phases. A suggested wording for the abstract: "Nifty and Bank Nifty changed their expiry calendar five times between 2023 and 2026; in the same period the expiry calendars of other NSE and BSE index options changed 17 times (Table 1)." The 17 are the rows of `calendar_changes.csv` for indices other than Nifty and Bank Nifty, excluding new listings and SEBI rows.
* **Minor 16.** See Section 1, Result 2. The Tuesday and Thursday single-weekly regimes differ in whole-day premium under every baseline (Mann-Whitney p 0.004 to 0.012). The difference does not survive a Bonferroni correction for 18 tests (0.08 to 0.21).
* **Minor 18.** See Section 2 for the wording.

## 7. Limits of this response

* **The BSE calendar is reported, not verified.** BSE's servers remain blocked from this machine. Every clean-baseline result depends on the Sensex, Bankex and Sensex 50 dates. The NSE-only version of each result is in the same files, labelled "no NSE expiry (verified)".
* **Two calendar details are assumed or reconstructed.** Bankex monthly expiries in November and December 2024 are an unsourced assumption. NSE's special Saturday sessions are not in our weekday files.
* **The SEBI phases of February and April 2025 rest on a secondary summary.** The circular was not re-read.
* **Other indices' expiries are treated as all-or-nothing.** Midcap Select's early 2023 expiries had almost no trading (about 0.00002 of Nifty's expiry turnover). Excluding them from a clean baseline is conservative.
* **Noise-robust estimators on 30 or 60 one-minute returns are imprecise.** The simulation shows that pre-averaging is about twice as noisy as one-minute RV. A null result with pre-averaging is weak evidence of no effect. Tick data would settle it.
* **Several new tests were chosen after the referee report.** These are the prediction-interval test, the wild cluster bootstrap, the trend-adjusted change, the raw two-day difference and the pooled regression. The wild cluster bootstrap over-rejects on the Friday rows, so its small p-values are not evidence.
* **One small error was found in the event study's calendar.** Its pre-October 2022 Thursday rule has one date (19 October 2022) on which the bhavcopies show no Nifty or Bank Nifty expiry.

## 8. How to reproduce

The minute bars and filtered bhavcopies are in `data/raw` (git-ignored). The full-bhavcopy extract is committed in `data/external/`; to rebuild it from NSE's archive, run with `--fetch-nse`.

```
OMP_NUM_THREADS=1 python experiments/referee_expiry.py          # about 4 minutes
OMP_NUM_THREADS=1 python -m pytest -q tests/test_referee_expiry.py
```

The script checks that it rebuilds the committed `weekly_relative.csv.gz` (65,192 of 65,192 rows), every row of `event_tests.csv` (1,308 rows), the published R2 means and the published negative-control shares. It reports each check in `results/referee_expiry_stdout.txt`. A table version of the main results is in `results/referee_expiry/summary.md`.

## What the expiry evidence now supports

On real NSE data, with NSE expiry dates verified from full bhavcopies and BSE dates reported but not verified, the expiry evidence supports a narrower claim than the paper makes. Trading clearly follows each expiry rule change. Since September 2025 Nifty's Tuesday expiry day has a larger last-hour variance than the rest of its week, against ordinary days without any other index's expiry and with a realized kernel (+0.35, p 0.0009 on a clean baseline). When Bank Nifty moved its expiry from Thursday to Wednesday in September 2023, the end-of-day variance moved with it, both without any baseline (+0.90 in the last 30 minutes) and in a regression that controls for every index's expiry (+0.61), and it keeps about two thirds of its size with a realized kernel. The evidence does not support a multi-weekly expiry premium in 2023 to 2024, which cannot be measured on a clean baseline; a Friday-based calibration of the placebo test; calling the 6-month event results significant at 5%, since every 6-month placebo p rests on at most 4 independent windows from March 2019 to August 2023 and the small-sample prediction test gives 0.06 to 0.08 for the best results (in 3-month windows only Nifty's Tuesday last hour at event d gets below 5%, with a prediction p of 0.009, an honest randomization p at its floor of 0.071 and a pre-trend p of 0.045); a switch effect at event d; or the claim that efficient-price variance, as opposed to noise that reverses within a minute, rises in the last half hour, because part of the one-minute end-of-day premium is microstructure noise of the bid-ask type that appears only on expiry days. "Expiry raises volatility at the end of the day" should therefore become "expiry days show more, and more quickly reversing, price movement in the last hour, most clearly at Bank Nifty's 2023 switch and on Nifty's Tuesday since 2025; how much of it is efficient-price variance cannot be settled with one-minute data".
