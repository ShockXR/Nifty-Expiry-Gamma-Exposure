# P96 pre-registration: did losing the weekly expiry remove the expiry-weekday volatility premium? (synthetic control)

**Status.** Written on 3 November 2026 (phase 1), **before any outcome was computed on real data for this study**
(no volatility, variance, premium, fit or treatment effect), and committed to git before phase 2 runs. Later
changes are listed as deviations in `docs/SYNTHETIC_CONTROL.md` (section 13). Phase 2 files: script
`experiments/synthetic_control.py` (subcommand `analyse --phase2`), tables in `results/synthetic_control/`, log
`results/synthetic_control_stdout.txt`, write-up `docs/SYNTHETIC_CONTROL.md`, tests on SYNTHETIC data in
`tests/test_synthetic_control.py`. All results will be REAL DATA.

**Prior knowledge.** I read `docs/EVENT_STUDY.md` and `docs/REAL_DATA_RESULTS.md` (R2), so the Bank Nifty
numbers quoted in section 1 were known to me. For phase 1 the real index files were read only to check which
indices exist on every day, which have listed derivatives, whether their prices are present and internally
consistent (High above Low, Open and Close inside the day's range), and that the downloaded files agree with the
filtered copies already in `data/raw/`. Expiry dates were read from the F&O bhavcopies. No range, variance or
premium of any index was computed, and none of the working files of this study contain such a computation.

## 0. Summary in plain English

On 20 November 2024 SEBI's rules ended weekly options on Bank Nifty, Fin Nifty and Nifty Midcap Select. Each of
these indices used to have an "expiry day" every week (Wednesday, Tuesday and Monday). We ask whether that
weekday was more volatile than the rest of the week because of the expiry, and whether the extra volatility went
away when the weekly options stopped.

The difficulty is that markets change for many reasons at once. So we build a **synthetic Bank Nifty**: a
weighted mix of 19 other NSE sector indices that had no listed options (Nifty Auto, Nifty IT, Nifty Pharma and
so on). The weights are chosen so that, before the rule change, the mix moved like Bank Nifty from week to week.
After the rule change the mix shows what Bank Nifty would probably have looked like without the change. The
**gap** between the real Bank Nifty and the synthetic one after the change is our estimate of the effect. We
predict the gap is **negative** (the expiry weekday became calmer).

To judge whether the gap is real or luck, we pretend, one at a time, that each of the 19 sector indices was the
one that changed, and build a synthetic version of it from the other 18. If Bank Nifty's change stands out more
than all 19 pretend cases, that happens by chance only 1 time in 20 (p = 0.05). That is the smallest p-value
this design can produce. Fin Nifty and Midcap Select are checked the same way as secondary questions.

Honest expectation: the measure comes from one high and one low price per day, which is noisy, and the
expiry-day premia this project has measured so far are small. Unless the synthetic control tracks Bank Nifty very
closely, the study can only detect a large change (section 11), so a "no support" result would most likely mean
"too imprecise to tell", not "no effect".

**Words used below.**

* **Week-relative value:** the log variance on one weekday minus the average log variance of the other trading
  days of the same week. Anything that makes the whole week volatile cancels.
* **Log points:** differences of logs. +0.30 means about 1.35 times; -0.30 means about 0.74 times.
* **Donor:** an index used to build the synthetic control. **J** is the number of donors (19).
* **Gap:** treated index minus its synthetic control, week by week.
* **RMSPE:** root mean squared gap (the typical size of the gap) over a set of weeks. **Pre RMSPE** measures how
  well the synthetic control fits before the change; **post RMSPE** how far apart they are after.
* **Placebo:** the same calculation for an index that was not treated. Its result shows what chance alone does.
* **Holm correction:** a rule that raises p-values when several questions are tested together, so that the
  chance of any false alarm stays at 5%.

## 1. Question, and what is already known

**Question.** SEBI's circular of 1 October 2024 (SEBI/HO/MRD/TPD-1/P/CIR/2024/132) allowed weekly index options
on only one benchmark index per exchange from 20 November 2024. On NSE, Nifty 50 kept its weekly options; Bank
Nifty (weekly expiry on Wednesday), Fin Nifty (Tuesday) and Nifty Midcap Select (Monday) lost theirs. **Did
losing the weekly expiry remove the volatility premium each of these indices carried on its old expiry weekday?**
Primary unit: Bank Nifty, Wednesday. Prediction: a negative gap.

**Already known (REAL DATA, published in this repository).** `docs/EVENT_STUDY.md` studied the same event (event
c) for Bank Nifty with 1-minute bars and Nifty 50 as the comparison:

* Bank Nifty's expiry-Wednesday premium from September 2023 to November 2024 (regime R1, 42 weeks) was +0.04
  (interval -0.12 to +0.21) for whole-day 5-minute realized variance, +0.15 (+0.06 to +0.23) for per-minute
  Rogers-Satchell and +0.32 (+0.09 to +0.55) for the last 30 minutes. From November 2024 to August 2025 (R2, 28
  weeks) the same Wednesday showed -0.13, -0.16 and -0.19 (Table 3).
* The pre-specified difference-in-differences (Bank Nifty Wednesday minus Nifty Thursday, 5-minute variance, 6
  months each side) was -0.01 (-0.61 to +0.55), and Bank Nifty's Wednesday alone changed by -0.17 (-0.50 to
  +0.17). Nifty's own Wednesday last 30 minutes also fell (-0.32, -0.72 to +0.07), so part of the change may have
  been market-wide (Table 4). Nifty is not a clean control, because the same SEBI package changed Nifty too.

So the treated series is partly known, from a different measure and different data. **What is new here:** the
comparison units are indices with no listed options at all, which the November 2024 package did not touch
directly; the measure is the daily range from NSE's own index file; and Fin Nifty and Midcap Select, for which
this project has no earlier result, are added. The Parkinson series of Bank Nifty, every donor series and every
gap are unknown at the time of writing.

## 2. Data

* **NSE daily index file** `ind_close_all_DDMMYYYY.csv` (Open, High, Low and Close for every NSE index). The
  copies in `data/raw/index_close/` were filtered on download and hold only Nifty 50, Nifty Bank and India VIX
  (944 files, 3 rows each; checked). The full files were therefore downloaded unmodified from NSE's archive
  server (`nsearchives.nseindia.com/content/indices/`) with `fetch` on 3 November 2026, for all 944 official
  trading days from 2 January 2023 to 2 November 2026 (no failed downloads). They hold 173 distinct index names.
* **NSE F&O bhavcopies.** The copies in `data/raw/bhavcopy/` hold only NIFTY and BANKNIFTY rows, so the full
  bhavcopies were downloaded for the 512 trading days from 1 August 2023 to 29 August 2025 and cut to index
  futures and index options rows (all symbols). They give the derivative symbols and the expiry dates of Fin
  Nifty and Midcap Select. They are used for calendars only.
* **Agreement check (REAL DATA).** All 2,832 rows (944 days x 3 indices) of `data/raw/index_close/` are present
  in the download with identical Open, High, Low and Close. Over the 512 common days, the NIFTY (109) and
  BANKNIFTY (78) expiry dates from the downloaded bhavcopies are identical to those from the repository's copies.
* **Data fingerprint.** SHA-256 over the downloaded files (definition in `data_fingerprint`):
  `49e64d24c9ddeddfc74351e4ee9eb00edf8877a3af61f911d8bdea69c791d223` (944 index files, 512 F&O files).
  The downloaded folder is not in git (about 82 MB); `fetch` rebuilds it, and `analyse` stops if the fingerprint
  of the files it is given differs.
* **Trading days.** The dates of `data/raw/index_daily.csv` (944), minus the one-hour Diwali (Muhurat) sessions
  of 1 November 2024 and 21 October 2025: 942 days. The study window, 4 September 2023 to 29 August 2025, has
  488 trading days.
* **Sensex calendar** (sensitivity S5 only): the reported weekday regimes in
  `data/external/sensex_expiry_calendar.csv`, turned into dates with the reported holiday rule (previous trading
  day). Reported, not verified (BSE files are not reachable). 104 days in the study window (65 Friday, 34
  Tuesday, 5 Thursday).

## 3. Donor rule and the donor pool (J = 19)

Applied to the 488 trading days of the study window, in this order (counts are REAL DATA from `coverage`):

| step | rule | indices left |
| --- | --- | --- |
| D0 | names seen on any study day (case and spacing ignored) | 148 |
| D1 | present on all 488 days | 102 |
| D2 | Open, High, Low and Close present and positive on every day, and High above Low on every day (all 68 also have Open and Close inside the range on every day) | 68 |
| D3 | sector or thematic index (not broad market, not a strategy or factor index, not bonds, VIX, leveraged, inverse or futures-based) | 25 |
| D4 | no listed index futures or options at any time in the window | 23 |
| D5 | no heavy overlap with a treated index (rule below) | **19** |

* **D2** removes, among others, 11 sector or thematic indices that are present every day but for which NSE's
  file gives only a closing value (Open, High and Low shown as "-") for part of the window, for example Nifty
  Housing, Nifty Mobility and Nifty India Defence. A range-based variance cannot be computed for them. No index
  failed D2 because of a day with High equal to Low.
* **D3** removes broad indices because they hold the same large banks and the same large stocks as the treated
  indices and include Nifty 50 itself; strategy indices are built from the same broad universe. The category is
  read from the name with fixed patterns (`classify_name`); a misclassification among broad, strategy and
  non-equity has no effect, since all three are removed.
* **D4.** Index-derivative symbols in the downloaded bhavcopies in the window: NIFTY, BANKNIFTY, FINNIFTY,
  MIDCPNIFTY and NIFTYNXT50 (from 24 April 2024). Among the 25 indices left after D3 this removes Nifty Bank and
  Nifty Financial Services (both treated, so never donors).
* **D5, overlap rule.** Historical constituent lists are not in the repository, so overlap is judged from the
  name: an index is removed if its name says it is made of banks, financial companies, insurers or
  capital-market companies, or is the services-sector index (which includes banks), or is made of midcap or
  small-cap stocks (the universe of Midcap Select). This removed Nifty Financial Services 25/50, Nifty PSU Bank,
  Nifty Private Bank and Nifty Services Sector. The name rule cannot see partial overlap (for example a thematic
  index that holds a few financial companies or midcaps); sensitivity S6, which drops the donor with the largest
  weight, is the guard against one contaminated donor driving the result. **One donor pool serves all three treated indices.** All
  three lost their weeklies on the same date, and every week-relative value uses the other weekdays as its
  baseline, so an index exposed to any treated index (for example a bank index, through Bank Nifty's Wednesday)
  is itself partly treated on every weekday.

**Donor pool (J = 19):** Nifty Auto, Nifty CPSE, Nifty Commodities, Nifty Consumer Durables, Nifty Energy, Nifty
FMCG, Nifty Healthcare Index, Nifty IT, Nifty India Consumption, Nifty India Digital, Nifty India Manufacturing,
Nifty Infrastructure, Nifty MNC, Nifty Media, Nifty Metal, Nifty Oil & Gas, Nifty PSE, Nifty Pharma, Nifty
Realty. All three treated indices and Nifty 50 pass D1 and D2. The full table is written by `coverage` as
`donor_table.csv`.

## 4. Variables

**Daily measure (primary: log Parkinson variance).** For index i on day d,
`v(i, d) = ln( (ln(High / Low))^2 / (4 ln 2) )` (Parkinson 1980). **Why Parkinson and not Garman-Klass.** Both use
NSE's official daily values. Garman-Klass (1980) adds a term in ln(Close / Open). NSE's official Open comes from
the pre-open call auction and its official Close from the closing-price rule (a weighted average over the last
half hour for the constituents in this period), so neither is a single traded index level, and the closing
average smooths exactly the last half hour in which this project found the expiry-day premium
(`docs/EVENT_STUDY.md`, finding 3). Parkinson depends only on the day's high and low, which are traded index
levels. (The description of NSE's opening and closing rules is general knowledge, not checked against an NSE
circular.) Garman-Klass is sensitivity S2. The log puts the measure on a percentage scale; its bias is roughly
constant from day to day for a given index and cancels in week-relative values.

**Weekly outcome.** For treated index T with old expiry weekday k (Monday 0 to Friday 4) and calendar week w:
`Y(i, w) = v(i, day k of week w) - mean of v(i, d) over the other trading days d of week w`, for the treated index
and for every donor, on the same weeks. A week counts only if day k is a trading day and at least 2 other trading
days exist.

**Weeks used** (for each treated index, the same weeks for all units): the Monday of the week lies in the window
(section 5); the week holds no monthly expiry and no holiday-shifted expiry of Nifty or of the treated index
(as in the event study; monthly weeks changed separately at event b, and Bank Nifty kept a monthly expiry after
November 2024); day k trades; at least 2 other trading days.

**Expiry calendars (REAL DATA, from the bhavcopies; an expiry is a day on which options expiring that day
traded, and it is monthly if an index future of the same symbol also expired that day; a weekday change needs
3 expiries on the new weekday):**

* Bank Nifty: weekly on Thursday to 24 August 2023 and on Wednesday from 6 September 2023; last weekly expiry
  13 November 2024 (20 November 2024 was a market holiday). Holiday-shifted weeklies (to Tuesday): 16 April, 30
  April, 16 July and 1 October 2024. Monthly: Thursday to 29 February 2024, Wednesday from 27 March 2024 (24
  December 2024 shifted to Tuesday), Thursday from 30 January 2025.
* Fin Nifty: weekly on Tuesday; last weekly expiry 19 November 2024. Holiday-shifted weeklies (to Monday): 14
  August, 18 September, 23 October and 13 November 2023. Monthly on Tuesday to 31 December 2024, Thursday from
  30 January 2025.
* Midcap Select: weekly on Wednesday to 16 August 2023 and on Monday from 21 August 2023; last weekly expiry 18
  November 2024. Holiday-shifted weeklies (to Friday): 29 September 2023, 17 May and 14 June 2024. Monthly on
  Monday to 30 December 2024 (shifted to Friday on 24 November and 22 December 2023 and 22 March 2024), Thursday
  from 30 January 2025.
* Nifty 50: weekly on Thursday throughout the study window (to 21 August 2025). Holiday-shifted (to Wednesday):
  10 April and 14 August 2024, 9 April and 30 April 2025.

The last weekly dates agree with the reported dates in `data/external/other_weekly_expiries.csv`.

## 5. Windows

| window | Bank Nifty (Wednesday) | Fin Nifty (Tuesday) | Midcap Select (Monday) |
| --- | --- | --- | --- |
| pre (fit) | 4 Sep 2023 to 27 Sep 2024: 56 weeks, **37 usable** | same: 56 weeks, **35 usable** | same: 56 weeks, **29 usable** |
| excluded gap | 30 Sep 2024 to 15 Nov 2024 | 30 Sep 2024 to 22 Nov 2024 | 30 Sep 2024 to 22 Nov 2024 |
| post (primary, 26 weeks) | 18 Nov 2024 to 16 May 2025: **17 usable** | 25 Nov 2024 to 23 May 2025: **17 usable** | 25 Nov 2024 to 23 May 2025: **15 usable** |
| post, long (S1) | to 29 Aug 2025: 28 usable | to 29 Aug 2025: 27 usable | to 29 Aug 2025: 25 usable |
| pre ending 26 Jul 2024 (S4) | 32 usable | 30 usable | 24 usable |
| placebo in time: fake pre / fake post | 23 / 14 | 19 / 16 | 17 / 12 |

(REAL DATA, calendar only, `week_counts.csv`.) Every usable pre week holds the treated index's own weekly expiry
on the target weekday (37 of 37, 35 of 35, 29 of 29), and no usable post week does. Bank Nifty's post window loses
8 weeks to monthly or shifted expiries and 1 week (18 November 2024) to the Wednesday holiday; its first usable
post week is the week of 2 December 2024.

* **Start of the pre-period: 4 September 2023**, the week of Bank Nifty's first Wednesday weekly expiry. Before it,
  Bank Nifty's Wednesday was an ordinary day; mixing those weeks in would make the fit target a blend of weeks
  with and without the premium under study, and the event study found that the end-of-day premium moved with
  the expiry in September 2023. Midcap Select's weekly had moved to Monday two weeks earlier (21 August 2023)
  and Fin Nifty's was on Tuesday throughout, so one start date suits all three.
* **End of the pre-period: 27 September 2024**, the Friday before the week of the circular (1 October 2024).
* **Circular-to-effective gap.** The weeks from 30 September 2024 to the week of each index's last weekly
  expiry are in neither period. After the circular, traders knew the weeklies would end, and weekly contracts
  kept trading until the last expiry, so these weeks are partly treated (anticipation and wind-down).
  Sensitivity S4 ends the pre-period on 26 July 2024, before SEBI's consultation paper of 30 July 2024 that first
  proposed the change (date from memory, not checked).
* **Post-period: 26 calendar weeks** from the Monday after the week of the index's last weekly expiry, fixed
  now. About six months, as in the event study's main window; S1 extends it to 29 August 2025, the last week
  before Nifty's weekly moved to Tuesday (event d).

## 6. Estimator

For one treated index, let y(t) be its weekly outcome and x_j(t) donor j's, for pre weeks t = 1 to T0 and post
weeks T0 + 1 to T0 + T1.

1. **Centre** every series on its own pre-period mean: y'(t) = y(t) - mean of y over pre weeks; the same for each
   x_j.
2. **Weights:** w = argmin of the sum over pre weeks of (y'(t) - sum_j w_j x'_j(t))^2, subject to every w_j >= 0
   and sum_j w_j = 1. Only pre-period outcomes are used (no other predictors). Solver (`simplex_weights`):
   non-negative least squares with the adding-up constraint as a heavily weighted extra row, polished with SLSQP;
   the better of the two is kept, weights below 1e-10 are set to 0.
3. **Gap:** g(t) = y'(t) - sum_j w_j x'_j(t). Its pre-period mean is 0 by construction.
4. **Estimate:** the average post-period gap, `effect = mean of g(t) over the T1 post weeks`. This equals the
   change in the treated index's premium from before to after, minus the same change for its synthetic control.
5. Also reported: the **pre-period premium over the synthetic control**, `premium_pre = mean over pre weeks of
   (y(t) - sum_j w_j x_j(t))` without centring. If the weekly expiry was the whole source of that premium, its
   full removal means effect = -premium_pre.

**Why the series are centred (an intercept-shifted synthetic control, as in Doudchenko and Imbens 2016 and Ferman
and Pinto 2021).** By design the treated index carries an expiry-day premium that no donor has, so before the
change its level is expected to lie above every donor. A plain weighted average of donors (classic synthetic
control, Abadie, Diamond and Hainmueller 2010) cannot reach a level outside the donors' range, and the poor fit
would be read as noise. Centring takes the level out of the fit, so the weights match the week-to-week movement,
and the estimate is the change in the gap, which is the question asked. The classic version (no centring;
estimate = mean post gap) is sensitivity S3.

What the estimate means: the effect on the treated index over and above any change on its old expiry weekday in
indices without options. If the expiry premium spilled over to the whole market, the synthetic control absorbs
that spillover, and the estimate understates the total effect.

## 7. Primary test and decision rule

**Placebo in space** (Abadie et al. 2010). The same estimator is run with each of the J = 19 donors as a fake
treated unit, fitted on the other 18 donors (a treated index is never a donor). For every unit u (Bank Nifty and
the 19 placebos), ratio(u) = post RMSPE / pre RMSPE.

**Primary p-value:** `p = (number of the 20 units whose ratio is at least Bank Nifty's) / 20`. The ratio test does
not use the direction of the gap; the direction enters the decision rule. Ties count against Bank Nifty.

**Decision (Bank Nifty, Wednesday, primary windows, log Parkinson, alpha = 0.05):**

* **Support:** effect < 0 and p <= 0.05 (Bank Nifty's ratio is the largest of the 20), and the placebo in time
  (section 9) does not also give p <= 0.05. If it does, the result is reported as "support, but the placebo in time
  also rejects (not reliable)".
* **Evidence against:** effect > 0 and p <= 0.05 (the old expiry weekday became more volatile relative to the
  synthetic control).
* **No support:** every other case, with one of three descriptions, checked in this order:
  (a) "no pre-period premium over the synthetic control to remove" if premium_pre <= 0;
  (b) "removal of the whole pre-period premium is ruled out" if -premium_pre lies below the 95% confidence set
  (section 9);
  (c) "too imprecise to tell" otherwise.

Under no effect the chance of a false "support" is about 2.5% (rank 1 of 20 has chance 1 in 20, and the sign is
negative about half of the time); the simulations of section 11 gave 2% to 4%.

## 8. Secondaries: Fin Nifty (Tuesday) and Midcap Select (Monday), with Holm

The same estimator, windows, test and donor pool, with each index's own target weekday and weeks. Their two
ratio p-values are adjusted together by Holm (1979); a secondary gets "support" only if its Holm-adjusted p is at
most 0.05 and its effect is negative ("evidence against" if positive). **Consequence of J = 19, stated now:** the
smallest possible raw p is 1/20, so the smallest possible Holm-adjusted p is 2 x 1/20 = 0.10. **Neither secondary
can reach 0.05**; both will be reported as estimates, ranks, raw and Holm p-values and confidence sets, and their
verdict will be "no support" whatever the data show. Secondaries cannot change the primary verdict.

**Joint test (supplementary, not a decision input).** The mean over the three treated indices of the
standardized effect (effect / pre RMSPE) is compared with the same mean for every combination that takes one
donor placebo from each index's table (19^3 = 6,859 combinations, in the spirit of Cavallo, Galiani, Noy and
Pantano 2013): one-sided p for a negative mean, smallest possible value 1/6,860.

## 9. Placebos, confidence set and cross-checks

* **Placebo in space:** section 7. Each placebo's effect, ratio, pre RMSPE and number of donors with weight are
  reported. Also reported: a one-sided rank p for the standardized effect (effect / pre RMSPE, prediction
  negative), which is more powerful against a steady shift (section 11) but is not the pre-registered decision
  statistic.
* **Placebo in time:** the whole analysis rerun with a fake change on 1 April 2024, using pre-period weeks only:
  fake pre 4 September 2023 to 31 March 2024, fake post 1 April to 27 September 2024 (26 weeks). Every treated
  index had its weekly expiry in both halves, so no effect is expected. Monthly weeks (the only weeks changed by
  event b in March 2024) are excluded anyway.
* **95% confidence set** (Firpo and Possebom 2018): for each constant effect a on a grid from -3.00 to +3.00 in
  steps of 0.01, subtract a from the treated index's post weeks, refit, and recompute its ratio p against the
  unchanged placebo ratios. The set is every a with p > 0.05. With J = 19 this is every a for which the treated
  index is not ranked first. Its ends, whether it is one unbroken interval, and whether it reaches the end of the
  grid (then the true set is wider) are reported.
* **Difference-in-differences cross-check:** treated minus the equal-weight average of the 19 donors; estimate =
  post mean minus pre mean of that difference. Reported with a Newey-West (4 lags) two-sided p and a one-sided
  rank p (each donor against the average of the other 18).
* **Nifty 50's Wednesday as a fake treated unit (descriptive only).** Nifty kept its weekly options (on
  Thursday), so it is not a donor. Its Wednesday, on Bank Nifty's weeks, shows whether a large index that kept
  its weekly options also moved against the donors on Wednesdays.

## 10. Sensitivity analyses (reported for all three indices; none changes a verdict)

* S1: long post window, to 29 August 2025.
* S2: log Garman-Klass variance, `ln( 0.5 (ln(H/L))^2 - (2 ln 2 - 1) (ln(C/O))^2 )`.
* S3: classic synthetic control (no centring; estimate = mean post gap).
* S4: pre-period ends 26 July 2024.
* S5: baseline days exclude Nifty expiry days and reported Sensex expiry days (Sensex moved from Friday to
  Tuesday in January 2025, inside the post window).
* S6: without the donor that gets the largest weight.
* S7: weeks need at least 3 other trading days.
* S8: baseline Thursday and Friday only. All three indices lost their weeklies on the same day, so each one's
  baseline contains the old expiry weekdays of the other two (and BSE's Bankex weekly was on Monday until
  November 2024, reported). If, for example, Fin Nifty's Tuesday expiry raised Bank Nifty's Tuesday more than the
  donors' Tuesday, its removal lowers Bank Nifty's baseline and biases the Bank Nifty gap upward (against the
  prediction). S8 removes Monday to Wednesday from every baseline.

## 11. Power and the minimum detectable effect implied by the design

**Granularity.** With J donors the smallest permutation p is 1/(J + 1). Here **J = 19, so the smallest p is 1/20 =
0.05**: the primary test can reject at 5% only when Bank Nifty's ratio is the largest of all 20 units. For the
secondaries the smallest Holm-adjusted p is 0.10 (section 8).

**Simulated power (SIMULATED data only, `power` subcommand, 400 simulated panels per effect size).** Each unit's
weekly value is a loading times a common factor plus independent noise; the effect is added to the treated
unit's post weeks and measured in units of that noise standard deviation (SD). Same J, T0 and T1 as the real
design:

Share of simulated panels in which the test rejects at 5% (simulation error about 0.02 either way). "Ratio
test" is the primary rule (rank 1 of 20 and a negative effect); "signed test" is the one-sided rank of the
standardized effect, reported but not a decision input. For Fin Nifty and Midcap Select the columns use raw p;
after Holm their power is zero (section 8). Weeks: Bank Nifty T0 = 37, T1 = 17; Fin Nifty 35, 17; Midcap Select
29, 15.

| effect (noise SD) | Bank Nifty, ratio test | Bank Nifty, signed test | Fin Nifty, ratio | Fin Nifty, signed | Midcap Select, ratio | Midcap Select, signed |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.02 | 0.04 | 0.03 | 0.06 | 0.04 | 0.07 |
| -0.25 | 0.10 | 0.21 | 0.06 | 0.17 | 0.06 | 0.12 |
| -0.5 | 0.12 | 0.34 | 0.11 | 0.36 | 0.08 | 0.29 |
| -0.75 | 0.23 | 0.66 | 0.21 | 0.64 | 0.18 | 0.58 |
| -1 | 0.39 | 0.88 | 0.29 | 0.83 | 0.27 | 0.79 |
| -1.25 | 0.58 | 0.97 | 0.56 | 0.95 | 0.46 | 0.93 |
| -1.5 | 0.67 | 1.00 | 0.69 | 0.97 | 0.62 | 0.98 |
| -1.75 | 0.86 | 1.00 | 0.83 | 1.00 | 0.78 | 0.99 |
| -2 | 0.93 | 0.99 | 0.92 | 1.00 | 0.85 | 1.00 |
| -2.5 | 0.98 | 1.00 | 0.99 | 1.00 | 0.95 | 1.00 |

**Minimum detectable effect with 80% power (MDE80), by linear interpolation in the table:**

| | Bank Nifty | Fin Nifty | Midcap Select |
| --- | --- | --- | --- |
| ratio test (primary rule), in noise SD | 1.67 | 1.70 | 1.82 |
| the same, as a multiple of the pre RMSPE | **1.93** | **1.96** | **2.23** |
| signed test (reported only), in noise SD | 0.91 | 0.96 | 1.02 |
| the same, as a multiple of the pre RMSPE | 1.06 | 1.10 | 1.25 |

The multiple of the pre RMSPE divides the MDE by the median pre RMSPE of the simulated fits under no effect
(0.86, 0.87 and 0.82 noise SD; below 1 because the weights are fitted to the same weeks). Phase 2 multiplies the
real pre RMSPE by these numbers to state the MDE in log points (column `mde80_ratio_test_log_points`). The ratio
test needs almost twice the effect of the signed test because a steady shift raises the post RMSPE only through
the square root of (noise variance + shift squared), while the ratio also moves with the luck of each fit. The
ratio test is kept as the primary statistic because it was fixed in the study design, does not assume a
direction, and down-weights placebos that fit badly before the change. Under no effect it gave a false "support"
in 2% to 4% of simulated panels.

**Scale in log points.** The noise SD of the real gap is not known before phase 2 (estimating it needs the
outcomes). For scale only: if an index were a constant-volatility random walk (Brownian motion), its daily log
Parkinson variance would have an SD of 0.582 and its week-relative value (4 other days) an SD of 0.655 log points
(SIMULATED, `brownian_noise_floor`). The real gap will be smaller to the extent that an index and its synthetic
control move together within each day, and larger to the extent that real volatility varies more than in that
model. As an illustration only: if the gap's noise SD were 0.655, the ratio test's MDE80 for Bank Nifty would
be about 1.67 x 0.655 = 1.1 log points (a factor of about 3 in variance), and the signed test's about 0.6. Every
whole-day expiry-day premium in the regime table of `docs/EVENT_STUDY.md` (Table 3, 5-minute variance and
Rogers-Satchell, both indices, 2019 to 2026) lies between -0.25 and +0.25 log points. **Unless the synthetic
control tracks the treated index much more closely than that illustration, the primary test can detect only a
change several times larger than any whole-day premium measured so far.** Phase 2 reports the MDE in log points
as the multiplier above times the observed pre RMSPE.

## 12. What is reported whatever happens

For each of Bank Nifty, Fin Nifty and Midcap Select (and Nifty 50's Wednesday, descriptive): the number of pre and
post weeks; the donor weights; the weekly series of the treated index, its synthetic control and the gap; the
effect, premium_pre, the post-period premium over the synthetic control, pre and post RMSPE and their ratio; the
ratio p, its rank among the 20 units, the one-sided standardized-effect p; the 95% confidence set; the
difference-in-differences cross-check with both p-values; the placebo in time; S1 to S8; Holm-adjusted p for the
secondaries; the joint test; the full placebo tables; a figure of all gaps; the verdicts; and the MDE in log
points. A null result is written up with the same detail as a positive one.

## 13. Deviations policy

* Nothing in sections 2 to 10 changes after this file is committed. Any change (including a bug fix) is listed in
  `docs/SYNTHETIC_CONTROL.md` under "Deviations" with its reason and date, and the pre-registered result is shown
  next to the changed one. The verdict always comes from the pre-registered specification.
* If a bug changes a number, the corrected pre-registered specification is the primary result, and the bug and
  both numbers are reported.
* If the data fingerprint differs from section 2, `analyse` stops. Running on other files needs
  `--accept-new-data` and is a deviation.
* If a donor or treated index turns out to have an unusable week that the coverage check missed, that week is
  dropped for all units (the code drops any week with a missing value) and the count is reported.
* Analyses added after seeing results are labelled post hoc and cannot change a verdict.

## 14. Reproducibility

```bash
OMP_NUM_THREADS=1 python experiments/synthetic_control.py fetch --out DIR        # download (needs network access)
OMP_NUM_THREADS=1 python experiments/synthetic_control.py coverage --full DIR --out DIR2   # phase 1, no outcomes
OMP_NUM_THREADS=1 python experiments/synthetic_control.py power --J 19 --T0 37 --T1 17 --effects 0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5 --n-sim 400
OMP_NUM_THREADS=1 python experiments/synthetic_control.py analyse --phase2 --full DIR      # phase 2 only
OMP_NUM_THREADS=1 python -m pytest -q tests/test_synthetic_control.py                     # SYNTHETIC data only
```

The analysis has no random parts (every placebo is computed). The power simulations use
`numpy.random.default_rng([20261102, 96, 0, k])` for effect number k; 20261102 is the project's master seed.
`run_info.json` records the data fingerprint and the library versions.

## Sources

1. SEBI circular SEBI/HO/MRD/TPD-1/P/CIR/2024/132, 1 October 2024 (listed as verified in `docs/LITERATURE_REVIEW.md`).
2. Last weekly expiry dates of Fin Nifty and Midcap Select as reported (NSE circular of 10 October 2024 as summarised by Zerodha): `data/external/other_weekly_expiries.csv`.

Cited from memory, not looked up:

3. Abadie, A., Diamond, A. and Hainmueller, J. (2010), "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program", Journal of the American Statistical Association 105(490), 493-505.
4. Abadie, A., Diamond, A. and Hainmueller, J. (2015), "Comparative Politics and the Synthetic Control Method", American Journal of Political Science 59(2), 495-510 (placebo in time).
5. Doudchenko, N. and Imbens, G. W. (2016), "Balancing, Regression, Difference-in-Differences and Synthetic Control Methods: A Synthesis", NBER Working Paper 22791.
6. Ferman, B. and Pinto, C. (2021), "Synthetic Controls with Imperfect Pretreatment Fit", Quantitative Economics 12(4), 1197-1221.
7. Firpo, S. and Possebom, V. (2018), "Synthetic Control Method: Inference, Sensitivity Analysis and Confidence Sets", Journal of Causal Inference 6(2).
8. Cavallo, E., Galiani, S., Noy, I. and Pantano, J. (2013), "Catastrophic Natural Disasters and Economic Growth", Review of Economics and Statistics 95(5), 1549-1561.
9. Parkinson, M. (1980), "The Extreme Value Method for Estimating the Variance of the Rate of Return", Journal of Business 53(1), 61-65.
10. Garman, M. B. and Klass, M. J. (1980), "On the Estimation of Security Price Volatilities from Historical Data", Journal of Business 53(1), 67-78.
11. Holm, S. (1979), "A Simple Sequentially Rejective Multiple Test Procedure", Scandinavian Journal of Statistics 6(2), 65-70.
12. Newey, W. K. and West, K. D. (1987), Econometrica 55(3), 703-708.
13. SEBI consultation paper on equity index derivatives, 30 July 2024 (date only).
