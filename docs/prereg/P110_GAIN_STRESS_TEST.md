# P110 pre-registration: do P106's claims about the hybrid's gain hold on Bank Nifty and Fin Nifty?

**Status.** Written on 3 November 2026 (phase 1) by an AI agent (Claude) working for Shreyan Thanki, at the request
of the project's lead agent. **This design is committed to git before any outcome of this study has been
computed.** No forecast of this study has been made on real data, and no loss, contribution, margin, test
statistic or bootstrap draw of this study has been computed on real data. The only real-data numbers used below
are committed results of earlier studies (mostly P106's, `results/p106_gain_decomposition/`) and simple arithmetic
on them (Section 7 says exactly which). The script and its tests were run end to end on simulated markets (SYNTHETIC). The commit that adds
this file is the time stamp; phase 2 (the run) starts only after that commit exists and has been pushed. The code
that phase 2 runs (`experiments/p110_gain_stress_test.py`) and its tests (`tests/test_p110_gain_stress_test.py`,
synthetic data only) are in the same commit. Phase 2 writes `results/p110_gain_stress_test/` and
`docs/P110_GAIN_STRESS_TEST.md`. All results will be REAL DATA.

## In plain words

Our hybrid forecasts how much an index will move over the next 15 or 30 minutes. It takes the anchor paper's
model (mcsGARCH) and multiplies its forecast by a correction learnt by LightGBM from 41 inputs. Study P106 refitted
the hybrid five times, each time without one group of inputs, and concluded three things:

1. **The gain comes from recent realized volatility and the time of day.** Removing either group makes the hybrid
   clearly worse.
2. **The option-implied inputs, India VIX and the calendar inputs add nothing.**
3. **A hybrid that sees only mcsGARCH's own inputs (the `mcsinfo` hybrid) gets the whole gain.**

These claims rest on five sets of random seeds on Nifty but on only one seed set on Bank Nifty and Fin Nifty, and
P106 never ran the `mcsinfo` hybrid on those two indices. Claim 2 also rests only on tests that failed to find an
effect. A test that fails to find an effect is not evidence that there is none: with few days or noisy data it
fails even when the effect is real.

This study does three things:

* It refits the hybrid, the five "without one group" versions and the `mcsinfo` hybrid on Bank Nifty and Fin
  Nifty with four more seed sets, at 15 and 30 minutes, changing nothing else.
* It tests claim 2 properly, with an **equivalence test**. We fix a margin in advance: 1 point of mcsGARCH's QLIKE
  (1% of mcsGARCH's average loss). The test asks whether the data can rule out that the group is worth 1 point or
  more, in either direction. Only if they can do we say the group "adds nothing".
* It tests claim 3 properly: does the `mcsinfo` hybrid beat mcsGARCH on each index, and is it **non-inferior** to
  the full hybrid (at most 1 point worse)?

**What is new here, and what is not.** The days are the same ones P106 used (660 on Bank Nifty, 604 on Fin Nifty),
and P106's seed-7 results on these two indices are public. So this study does not bring new market data. The new
evidence is: four new seed sets on these two indices, the `mcsinfo` hybrid on these two indices (never run before),
and the equivalence and non-inferiority tests. To keep the already-public seed-7 results out of the main tests,
**the primary tests use the daily losses averaged over the four new seed sets only**; seed set 7 enters only the
"every seed set agrees" rule and a robustness row.

## Why this study

P106 (`docs/P106_GAIN_DECOMPOSITION.md`) changed how the project describes its forecasting result: the gain is a
learnt correction of mcsGARCH using mcsGARCH's own kind of information, not option-market information. Its section
13 says any sentence crediting the option-implied features with the gain should go. Before that wording reaches the
paper and the professor, it should be checked where P106 was thinnest: on the other two indices, with more than one
seed set, and with tests that can actually support "adds nothing".

## Prior knowledge (what was known when this was written)

All REAL DATA, from committed files. "Points" means points of mcsGARCH's mean daily QLIKE (1 point = 1% of it).

* **The hybrid's gain over mcsGARCH** (P106 `contributions.csv`, seed set 7; P91 and P104 runs): Bank Nifty 3.92
  points at 15 minutes and 5.03 at 30 minutes (660 test days, 10 August 2023 to 22 April 2026); Fin Nifty 4.90 and
  6.09 (604 test days, 10 August 2023 to 27 January 2026); Nifty 5.80 and 7.65 (792 test days).
* **P106's seed-7 group contributions on these two indices** (`tests.csv` rows `S5_*` and `S6_*`; positive =
  removing the group raises the loss; one-sided DM p; Holm over 10 rows per index):

  | Index | Group | 15 min: contribution (p, Holm p) | 30 min: contribution (p, Holm p) |
  |---|---|---|---|
  | Bank Nifty | realized | +1.58 (below 0.001, below 0.001) | +1.79 (below 0.001, below 0.001) |
  | Bank Nifty | time of day | +0.88 (below 0.001, below 0.001) | +0.95 (below 0.001, below 0.001) |
  | Bank Nifty | implied | -0.21 (0.788, 1.000) | -0.48 (0.884, 1.000) |
  | Bank Nifty | VIX | -0.06 (0.720, 1.000) | -0.24 (0.975, 1.000) |
  | Bank Nifty | calendar | -0.02 (0.581, 1.000) | +0.15 (0.192, 1.000) |
  | Fin Nifty | realized | +1.94 (below 0.001, below 0.001) | +1.95 (below 0.001, below 0.001) |
  | Fin Nifty | time of day | +1.62 (below 0.001, below 0.001) | +1.57 (below 0.001, below 0.001) |
  | Fin Nifty | implied | +0.20 (0.228, 0.911) | +0.17 (0.350, 0.911) |
  | Fin Nifty | VIX | +0.15 (0.031, 0.156) | +0.09 (0.234, 0.911) |
  | Fin Nifty | calendar | +0.20 (0.006, 0.035) | +0.06 (0.328, 0.911) |

  So on Fin Nifty at 15 minutes the calendar group already *helped* a little with seed set 7 (Holm 0.035), the
  other way round from Nifty.
* **Not run anywhere except Nifty:** the `mcsinfo` hybrid and any seed set other than 7. On Nifty (P106 `S3_*`
  rows, seed set 7) the `mcsinfo` hybrid was 6.06% and 8.01% better than mcsGARCH, and 0.26 and 0.36 points
  *better* than the full hybrid; in all five seed sets it had the lower loss (seed means 0.28 and 0.22 points).
* **Seed noise.** P99 (`results/seed_bootstrap/summary.csv`, Nifty): the standard deviation of the hybrid's margin
  across seed ensembles is 0.075 points at 15 minutes and 0.112 at 30 minutes. On Nifty the seed SD of a group's
  contribution across P106's five seed sets was 0.03 to 0.26 points (`verdict.csv`). The P91 and P104 seed checks
  (three single seeds at 30 minutes) found gaps of up to 0.18 points (Bank Nifty) and 0.56 points (Fin Nifty)
  between single seeds; the registered forecasts average five seeds, which shrinks this.
* **Days that dominate cross-index comparisons.** P107 (`posthoc_extreme_days.csv`) found that 4 June 2024 (the
  general-election result day) is among the two most extreme days of the hybrid's advantage at 15 and 30 minutes.

## 1. Variants and seeds

**Variants** (P106's definitions, unchanged; `DROP_SETS` in `experiments/p106_gain_decomposition.py`):

| Variant | Columns removed from the 41 | Model name |
|---|---|---|
| `full` | none | `hybrid_base` |
| `implied` | the 16 option-implied columns | `hybrid_drop_implied` |
| `vix` | `log_vix` | `hybrid_drop_vix` |
| `intraday` | the 4 time-of-day columns | `hybrid_drop_intraday` |
| `calendar` | the 8 calendar and regime columns | `hybrid_drop_calendar` |
| `realized` | the 12 realized-history columns (including mcsGARCH's own forecast) | `hybrid_drop_realized` |
| `mcsinfo` | implied, VIX and the calendar group except `is_expiry` (24 columns) | `hybrid_drop_mcsinfo` |

`optionmkt`, `implied_level` and `implied_shape` are not run (not asked; P106 found nothing in them on Nifty).

**Seed sets.** Base seeds 7 (the headline), 101, 102, 103 and 104 (the first four of P99's new ensembles, the ones
P106 used on Nifty). Each base seed s gives LightGBM seeds s, s + 1000, ..., s + 4000 (five-seed averaging) and also
seeds the pass that picks the number of boosting rounds, exactly as in P99 and P106.

## 2. Design

**What is refitted.** Exactly P106's walk-forward on each index (`run_walkforward` with the committed
`WalkForwardConfig`: 150 training days, refit before every block of 21 trading days, mcsGARCH refitted every 21
days, LightGBM default hyperparameters with the number of rounds chosen inside each training window, five-seed
averaging). Nothing is retuned. The script reuses P106's code for the configuration, the job, the work files and
the settings check (it refuses to run if any setting, the number of days or the last day differs from the committed
run of that index).

**What is run** (one process; all seven variants share one mcsGARCH fit per job):

| Index | Days loaded | Test days | Targets | Seed sets | Variants | Jobs | Variant fits |
|---|---|---|---|---|---|---|---|
| Bank Nifty | 810 (2 January 2023 to 22 April 2026) | 660 (10 August 2023 to 22 April 2026) | next 15, next 30 | 7, 101, 102, 103, 104 | the 7 above | 10 | 70 |
| Fin Nifty | 754 (2 January 2023 to 27 January 2026) | 604 (10 August 2023 to 27 January 2026) | next 15, next 30 | 7, 101, 102, 103, 104 | the 7 above | 10 | 70 |

Seed set 7 is rerun for all seven variants (not copied from P106's git-ignored work files) so that every forecast
this study uses is made by this study's commands; Section 9 checks that the rerun is identical to P106's files.

**Nifty is not refitted.** Where Nifty enters (secondary families S4a and S5), the daily losses are P106's committed
files `results/p106_gain_decomposition/daily_losses_nifty_<target>_s<seed>.csv`.

**Samples.**

* **Primary sample:** for each index and target, each model's daily mean QLIKE **averaged over the four new seed
  sets (101 to 104)**. mcsGARCH and `mcsgarch_iv` are deterministic and must be identical in every seed set (checked).
  The average estimates the expected loss of one five-seed ensemble, integrating over LightGBM's randomness; it
  leaves out seed set 7 because its results on these indices are already public.
* Seed set 7 alone, all five seed sets averaged, and each seed set alone are robustness or descriptive samples.

## 3. Statistics

All losses are QLIKE, y/f - log(y/f) - 1, averaged within each day first (P106's `day_frame`). L_x is the mean over
days of model x's daily QLIKE; "points" are 100 x (difference) / L_mcs.

* **Contribution of group v:** C_v = 100 x (L_v - L_full) / L_mcs. Positive: removing the group raises the loss, so
  the group helps. Same definition as P106.
* **`mcsinfo` against mcsGARCH:** M = 100 x (L_mcsinfo - L_mcs) / L_mcs (negative: `mcsinfo` is better).
* **`mcsinfo` against the full hybrid:** N = 100 x (L_mcsinfo - L_full) / L_mcs (positive: the full hybrid is better).

**Tests.** All are built on the project's Diebold-Mariano (DM) test of a daily series d_t
(`niftygex.evaluation.diebold_mariano`: Newey-West variance with bandwidth floor(4 (n/100)^(2/9)), the
Harvey-Leybourne-Newbold factor, Student t with n - 1 degrees of freedom).

* **"Removing the group raises the loss"** (carriers, and S1): one-sided DM test of L_v,t against L_full,t, exactly as
  P106 (`contribution`).
* **"`mcsinfo` beats mcsGARCH":** one-sided DM test of L_mcs,t against L_mcsinfo,t, exactly as P106's S3 (b).
* **Equivalence, "the group adds nothing" (TOST, two one-sided tests).** With d_t = 100 x (L_v,t - L_full,t) /
  L_mcs, whose mean is C_v, and margin delta:
  * test 1: H0 C_v >= delta against C_v < delta, the DM test of d_t - delta with the alternative "less";
  * test 2: H0 C_v <= -delta against C_v > -delta, the DM test of d_t + delta with the alternative "greater";
  * the TOST p-value is the larger of the two; it is below 0.05 exactly when the 90% confidence interval of C_v lies
    inside (-delta, +delta). The interval and "the smallest margin that would pass" (the larger end of the interval
    in absolute value) are reported for every row, so a reader can apply another margin.
* **Non-inferiority of `mcsinfo` to the full hybrid:** H0 N >= delta against N < delta (test 1 above with
  `mcsinfo` in place of the group).
* L_mcs, the scale of the points, is treated as fixed in these tests. Robustness R3 removes this simplification:
  the same tests on the ratio C_v with a delta-method Newey-West variance (P106's `ratio_var`).

**The margin: delta = 1 point of mcsGARCH's QLIKE** (that is, 1% of mcsGARCH's mean daily QLIKE on that index and
horizon). Reasons, in order of weight:

1. **P106 itself used 1 point as the size that matters.** Its registered expectation table said the
   option-implied group's "adds nothing" would be contradicted if it were supported "with more than 1 point". This
   study asks whether the data can rule out exactly that.
2. **It is far above seed noise.** The seed SD of the hybrid's margin is about 0.08 to 0.11 points (P99), and of a
   group's contribution 0.03 to 0.26 points on Nifty (P106), so a 1-point effect cannot be made by the seeds alone.
3. **It is a meaningful part of the gain.** The full hybrid's gain is 3.92 to 6.09 points on these two indices, so
   1 point is 16% to 26% of it. A group that is worth less than 1 point carries at most about a quarter of the gain.
4. The margin was suggested in the lead agent's brief ("for example ... 1 point of mcsGARCH's QLIKE") before any
   power figure in Section 7 was computed. The power figures did not change it.

**Its weakness, stated now.** On Bank Nifty the time of day, which P106 counts as one of the two groups that carry the
gain, was worth only 0.88 and 0.95 points with seed set 7. So "equivalent to zero within 1 point" can hold for a group
about as important as Bank Nifty's time of day. For this reason S3 repeats every equivalence and non-inferiority
test with a margin of 0.5 points, and every row reports the smallest margin that would pass.

**Seed agreement.** For each row, the estimate is also computed in each of the five seed sets separately. A seed set
"agrees" with a row if its estimate points the way the row's claim does: C_v > 0 for "removing raises the loss";
-1 < C_v < 1 for equivalence; N < 1 for non-inferiority; M < 0 for "beats mcsGARCH".

**Pooled versions** (secondary S4): for each index, the daily series in points of that index's own mcsGARCH QLIKE,
on the dates common to the indices pooled; their equal-weight average, with P107's Driscoll-Kraay covariance
(`index_means_dk` in `experiments/p107_pooled_test.py`, which allows any correlation between the indices on the same
day and serial correlation over neighbouring days), the HLN factor and Student t with T - 1 degrees of freedom. The
same one-sided, TOST and non-inferiority tests as above. Three-index pool: Nifty, Bank Nifty and Fin Nifty on their
common test days (604, to 27 January 2026, as in P107); two-index pool: Bank Nifty and Fin Nifty on their common test
days.

**Period test** (exploratory E1): P106's two-sided delta-method test (`period_diff_test`) of whether N, the value of
the information mcsGARCH lacks, differs between test days to 19 November 2024 and days from 20 November 2024.

## 4. Hypotheses, families and tests

There are three primary families, one for each of P106's claims, each with its own Holm adjustment. Each family
answers a different question and leads to its own verdict; robustness R6 also adjusts all 28 primary p-values
together, and any verdict that differs under R6 will be reported next to it.

**F1, claim 1: the realized history and the time of day carry the gain (8 tests).** For v in {realized, time of
day}, index in {Bank Nifty, Fin Nifty}, horizon in {15, 30 minutes}: H1: C_v > 0. One-sided DM. Holm over 8.

**F2, claim 2: the option-implied, VIX and calendar groups add nothing (12 tests).** For v in {implied, VIX,
calendar}, each index, each horizon: H1: -1 < C_v < 1. TOST, margin 1 point. Holm over 12.

**F3, claim 3: the `mcsinfo` hybrid gets the gain (8 tests).** For each index and horizon: (a) H1: M < 0 (it beats
mcsGARCH), one-sided DM; (b) H1: N < 1 (non-inferior to the full hybrid within 1 point). Holm over 8.

**Verdict of each primary row** (fixed now):

| Row type | Passes if | Label when it passes | Label when it does not |
|---|---|---|---|
| F1 | Holm p below 0.05 **and** all five seed sets have C_v > 0 | supported | not supported (seed-fragile if only the seed rule fails) |
| F2 | Holm TOST p below 0.05 **and** all five seed sets have -1 < C_v < 1 | equivalent | not shown (seed-fragile if only the seed rule fails) |
| F3 (a) | Holm p below 0.05 **and** all five seed sets have M < 0 | supported | not supported (or seed-fragile) |
| F3 (b) | Holm p below 0.05 **and** all five seed sets have N < 1 | non-inferior | not shown (or seed-fragile) |

Rows of F1 (and S1, S2) whose estimate is smaller in absolute value than twice the SD across the five seed sets are
also flagged "within seed noise", as in P106.

**Secondary families** (each with its own Holm adjustment; they cannot change a primary verdict, but S1 and S2 enter
the claim-level rules of Section 5):

* **S1 (12 tests), "the group adds something":** for implied, VIX and calendar, each index and horizon, H1: C_v > 0,
  one-sided DM (P106's test, now with the new seeds). This is the test that could show the opposite of claim 2.
* **S2 (4 tests), "the information mcsGARCH lacks adds something":** H1: N > 0 (the full hybrid beats `mcsinfo`),
  one-sided DM, each index and horizon (P106's S3 (c) on these indices).
* **S3 (16 tests), margin 0.5 points:** the 12 F2 tests and the 4 F3 (b) tests with delta = 0.5.
* **S4a (14 tests), pooled over Nifty, Bank Nifty and Fin Nifty**, and **S4b (14 tests), pooled over Bank Nifty
  and Fin Nifty:** at each horizon, realized and time of day (one-sided), implied, VIX and calendar (TOST, 1 point),
  `mcsinfo` beats mcsGARCH (one-sided) and `mcsinfo` non-inferior (1 point). Nifty's part uses P106's committed
  Nifty daily losses, averaged over seed sets 101 to 104.
* **S5 (8 tests), Nifty re-analysis:** the 6 TOST tests (implied, VIX, calendar at 15 and 30 minutes) and the 2
  non-inferiority tests of `mcsinfo`, on P106's committed Nifty daily losses averaged over seed sets 101 to 104 (792
  days). These data are public and P106's R1 rows already show their DM statistics, so this is a re-analysis with a
  new test, not new evidence.

**Robustness** (each with its own Holm adjustment, over the rows listed; descriptive with respect to the verdicts):

* **R1 (28):** the 28 primary rows on seed set 7 alone (P106's sample; for F1 and F2 this re-derives public numbers,
  for F3 it is new).
* **R2 (28):** the 28 primary rows on the daily losses averaged over all five seed sets.
* **R3 (16):** the 12 TOST rows and the 4 non-inferiority rows with the delta-method variance of the ratio.
* **R4 (28):** the 28 primary rows without 4 and 5 June 2024 (the general-election result day and the day after).
* **R5 (28):** the 28 primary rows without, on each index and target, the 1% of test days (6 days) with the largest
  mcsGARCH daily QLIKE. The rule uses the benchmark's loss only, not any comparison of the variants.
* **R6 (28):** the 28 primary p-values with one Holm adjustment over all of them.

**Placebo** (Holm within each family): **PL1 (16)**, the full hybrid with seed set 101, 102, 103 or 104 against the
same hybrid with seed set 7, TOST with the 1-point margin (only the seeds differ, so every row should pass: a check
that the equivalence test can say "nothing" when nothing is there). **PL2 (16)**, the same pairs, one-sided DM
(seed set 7 better), as P106's PL1 (expected: nothing significant).

**Exploratory** (no adjustment): **E1 (4)**, the period change of N on each index and horizon (P106 found on Nifty
that the information mcsGARCH lacks was worth more before 20 November 2024 than after, by 1.44 and 2.52 points). No
direction is stated for these two indices. Descriptive tables: each group's contribution in every sample and seed
set, the share of the gain with a 95% stationary-bootstrap interval (P106's `share_ci`, 10,000 draws, mean block 10
days), the period values of every contribution, the Spearman rank correlation of the five contributions with
Nifty's.

Every p-value goes in `results/p110_gain_stress_test/tests.csv` with P106's columns (test_id, hypothesis, role,
sample, estimate, statistic, p_raw, sidedness, p_method, p_adjusted, adjust_method). Any test added after this commit
is labelled "post hoc". For TOST rows the statistic is that of the binding (larger-p) side.

## 5. Claim-level decision rules, and what would contradict P106

For each index separately (Bank Nifty, Fin Nifty), using the primary verdicts and the S1 and S2 rows. "S1
significant" means Holm-adjusted p below 0.05 within S1 **and** all five seed sets with C_v > 0; the same for S2 with
N > 0.

**Claim 1 (realized history and time of day carry the gain):**

* **survives:** all four F1 rows of the index are "supported";
* **does not survive:** for the realized group or for the time of day, neither horizon is "supported";
* **partly survives:** anything else.

**Claim 2 (the option-implied, VIX and calendar groups add nothing):** first, for each group and horizon:

* **contradicted** if S1 is significant and the F2 row is not "equivalent" (the group helps, possibly by 1 point or
  more);
* **adds a little** if S1 is significant and the F2 row is "equivalent" (the group helps, but by less than 1 point);
* **shown** if the F2 row is "equivalent" and S1 is not significant;
* **not shown** otherwise (no contribution found, but 1 point cannot be ruled out).

Then, for the index: **contradicted** if any cell is contradicted; otherwise **qualified** if any cell is "adds a
little"; otherwise **survives (shown by equivalence)** if all six cells are "shown"; otherwise **survives only
weakly** (no group was found to add anything, but equivalence was not shown for every group and horizon).

**Claim 3 (the `mcsinfo` hybrid gets the whole gain):**

* **contradicted** if F3 (a) is not "supported" at either horizon, or if at some horizon S2 is significant and F3
  (b) is not "non-inferior" (the information mcsGARCH lacks is worth possibly 1 point or more there);
* **survives** if, at both horizons, F3 (a) is "supported" and F3 (b) is "non-inferior", and S2 is not significant
  at either horizon;
* **partly survives** otherwise (for example non-inferiority not shown at one horizon, or S2 significant but within
  the margin).

**P106's claims survive on an index** if claim 1 survives, claim 2 survives (shown or weakly) and claim 3 survives.
The write-up will say, for each index and each claim, which label applies and why, and it will reword P106's
sentences where needed.

**What would contradict P106, in short:** on either index, a realized or time-of-day contribution that is not
supported at either horizon; an option-implied, VIX or calendar contribution that is significantly positive with
every seed set agreeing and whose 90% interval reaches 1 point; or a `mcsinfo` hybrid that does not beat mcsGARCH,
or that is significantly worse than the full hybrid without being within 1 point of it.

## 6. What we expect (stated in advance; it changes no rule)

| Row | Expectation | Reason |
|---|---|---|
| F1 realized, time of day | supported on both indices at both horizons | seed set 7 gave DM statistics of 3.7 to 5.9 (all p below 0.001); the seed SD on Nifty was at most 0.26 points |
| F2 VIX and calendar | equivalent on both indices at both horizons | seed-7 90% intervals already end within 0.20 to 0.45 points of zero (Section 7) |
| F2 implied, 15 minutes | equivalent on both indices | seed-7 intervals end at 0.64 and 0.65 points |
| F2 implied, 30 minutes | **may not be shown**, at least on Bank Nifty | its seed-7 interval reaches 1.13 points on Bank Nifty and 0.88 on Fin Nifty; the step-down Holm makes the last rows hardest to pass |
| S1 | Fin Nifty calendar at 15 minutes may be significant (it was with seed set 7, Holm 0.035); nothing else | if so, claim 2 on Fin Nifty would be "qualified" (it adds a little), since its seed-7 interval was 0.07 to 0.33 points |
| F3 (a) | supported everywhere | on Nifty `mcsinfo` was 6.06% and 8.01% better than mcsGARCH, more than the full hybrid |
| F3 (b), S2 | non-inferior; S2 not significant | on Nifty `mcsinfo` was a little better than the full hybrid in all five seed sets |
| S5 (Nifty) | all equivalent and non-inferior at 1 point | P106's five-seed-average intervals (R1 rows) end within 0.11 to 0.76 points of zero; seed-7 non-inferiority bounds 0.25 and 0.34 (Section 7) |
| PL1, PL2 | PL1 all pass, PL2 none | only the seeds differ |

## 7. Power, from P106's public seed-7 numbers (computed before this file was committed)

To judge the margin's reach, the agent computed, from P106's committed `tests.csv` only, the standard error that each
seed-7 row implies: SE = estimate x sqrt((n - 1)/n) / DM statistic (the DM statistic is the estimate divided by its
standard error, times the HLN factor). From it, the 90% interval estimate +/- t(0.95, n - 1) x SE / sqrt((n - 1)/n)
and the smallest margin that interval would pass. These are P106's numbers rearranged; no new outcome is involved.
REAL DATA, seed set 7, points:

| Index | Group | 15 min: SE, smallest passing margin | 30 min: SE, smallest passing margin |
|---|---|---|---|
| Bank Nifty | implied | 0.26, 0.64 | 0.40, 1.13 |
| Bank Nifty | VIX | 0.11, 0.24 | 0.13, 0.45 |
| Bank Nifty | calendar | 0.11, 0.20 | 0.17, 0.43 |
| Fin Nifty | implied | 0.27, 0.65 | 0.43, 0.88 |
| Fin Nifty | VIX | 0.08, 0.28 | 0.12, 0.29 |
| Fin Nifty | calendar | 0.08, 0.33 | 0.13, 0.27 |

On Nifty, averaging over five seed sets lowered these standard errors (for VIX from 0.13 to 0.06 at 15 minutes,
for the implied group from 0.46 to 0.39; P106 `R1_*` rows), so the new-seed average should be at least as precise.
With 12 rows in F2, Holm compares the smallest TOST p with 0.05/12 = 0.0042 and the largest with 0.05; a row
whose interval ends near 0.9 points will pass only if the rows before it pass. **The implied group at 30 minutes is
the row most likely to stay "not shown"; the plan accepts this and the write-up will report it plainly.** With a
margin of 0.5 points (S3), the implied rows have almost no chance (their intervals alone are about +/- 0.4 to 0.7
points wide), which is why 0.5 is a sensitivity check and not the primary margin. Nothing similar can be computed
for `mcsinfo` on these indices (never run). On Nifty its non-inferiority bound (the upper end of the 90% interval
of N) was 0.25 and 0.34 points with seed set 7 (P106 `S3_*_mcsinfo` rows), well inside 1 point; its standard
errors there (0.31 and 0.42 points) are close to the implied group's, so F3 (b) should pass unless `mcsinfo` is
clearly worse than the full hybrid on these indices.

## 8. The exact commands

From the repository root, at the commit that adds this file, with the raw data listed in `data/MANIFEST.csv` in
`data/raw`:

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p110_gain_stress_test.py run --index BANKNIFTY \
    > results/p110_gain_stress_test/run_stdout_banknifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p110_gain_stress_test.py run --index FINNIFTY \
    > results/p110_gain_stress_test/run_stdout_finnifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p110_gain_stress_test.py analyze --plan-commit <this commit> \
    > results/p110_gain_stress_test/analyze_stdout.txt 2>&1
```

`run` uses one process, stores each variant's forecasts in `data/processed/p110/` (git-ignored; a finished job is not
recomputed if the run must be restarted) and the SHA-256 of every raw input. `analyze` writes `tests.csv`,
`tests_detail.csv`, `verdict.csv`, `claims.csv`, `equivalence_bounds.csv`, `contributions.csv`, `seed_repeats.csv`,
`shares.csv`, `periods.csv`, `pooled.csv`, `other_index_ranking.csv`, `reproduction_check.csv`,
`daily_losses_<index>_<target>_s<seed>.csv` (each variant's daily QLIKE, from which every number can be recomputed),
`timing.csv` and `run_info.json`.

**Expected run time.** P106's Bank Nifty and Fin Nifty jobs took about 55 to 85 seconds per variant fit
(`results/p106_gain_decomposition/timing.csv`). With 70 fits per index on one process, plus about 4 minutes to load
each market, we expect 1 to 1.75 hours per index, 2 to 3.5 hours in all.

## 9. Checks inside the run

* The full hybrid with seed set 7 must reproduce the committed `hybrid_base` forecasts of
  `results/pipeline_nse_banknifty/` and `results/pipeline_nse_finnifty/` exactly, with `y`, `mcsgarch` and
  `mcsgarch_iv` (`reproduction_check.csv`; no loss involved).
* The five group variants with seed set 7 must equal P106's work files (if those git-ignored files are still on
  disk) and must reproduce P106's committed `S5_*` and `S6_*` estimates and DM statistics.
* `y`, `mcsgarch` and `mcsgarch_iv` must be identical in every work file of an index and target, across variants and
  seed sets (the script refuses otherwise).
* Every raw file read must agree with `data/MANIFEST.csv` (recorded in `run_info.json`).

## 10. What will be reported, whatever the result

In `docs/P110_GAIN_STRESS_TEST.md`, for a beginner reader, every number labelled REAL DATA and taken from a file in
`results/p110_gain_stress_test/`: the answer first (does each of P106's claims survive on each index); the primary
table with verdicts and seed agreement; the equivalence intervals and the smallest passing margins; the S1 and S2
results; the 0.5-point margin; the pooled results; the Nifty re-analysis; the robustness, placebo and exploratory
results with their labels; the reproduction checks; the run time; a "Deviations from the plan" section; caveats; and
what it changes in the project's wording of P106.

## 11. Deviations policy

* After this file is committed the design does not change. Any change is a deviation, listed with its reason and
  with whether any outcome had been seen when it was decided.
* A crash is fixed with the smallest change that lets the run finish, and only if it changes no number of a job
  that already finished.
* If a reproduction check fails, that is reported, the cause is looked for, and the results are still given with
  the failure stated next to them.
* If a job cannot be run (time, crash), the rows that need it are reported as "not computed" with the reason. If
  time runs short, Bank Nifty is finished first; Fin Nifty rows would then be "not computed".

## 12. Known weaknesses

1. **No new days.** All tests use days P106 used, and P106's seed-7 results on them are public. New seeds remove one
   source of chance (LightGBM's randomness) but not the other (which days happened). The study cannot replace a test
   on later data.
2. **The margin is a choice.** 1 point is justified above, but it is not small relative to Bank Nifty's time-of-day
   contribution (Section 3). The 0.5-point rows and the reported intervals let a reader use another margin.
3. **Leave-one-group-out measures what a group adds given the others.** A group can look useless because another
   stands in for it (the chain and VIX; the realized history and mcsGARCH's own forecast). The `mcsinfo` variant
   removes the option market, VIX and most of the calendar together, which covers the stand-in problem for those
   three groups jointly, but not between the realized history and the time of day.
4. **The protocol is fixed.** Each variant keeps the headline's LightGBM settings; a tuned variant might do better.
5. **Five seed sets** show whether a sign is stable; they cannot estimate seed noise precisely, and the DM, TOST and
   period tests do not include seed noise in their p-values (P106's placebo showed this matters for the period test).
6. **The three indices move together.** Bank Nifty and Fin Nifty share news, VIX and much of their constituents
   (P107: the three are worth about two independent indices), so agreement on two indices is less than two
   independent confirmations.
7. **The scale of the points is treated as known** in the main tests (R3 relaxes it).
8. **Holm within each claim's family** rather than over all 28 primary tests is a choice; R6 shows the stricter
   version.

## 13. Code

* `experiments/p110_gain_stress_test.py`, new: `run` (forecasts, one invocation per index, one process, P106's
  `run_job`, `make_config`, work files and settings check) and `analyze` (Sections 3 to 5; P106's `contribution`,
  `ratio_var`, `period_diff_test`, `share_ci` and `holm`; P107's `index_means_dk`). `--synthetic` runs both stages on
  three simulated markets (labelled SYNTHETIC).
* No change to `src/` or to P106's or P107's files.
* `tests/test_p110_gain_stress_test.py`, new, synthetic data only (12 tests): the variants, groups, seeds and settings;
  the job plan; the DM pieces reproduce the project's DM test; the TOST equals the two shifted DM tests and the 90%
  interval rule; size of the TOST and of the non-inferiority test at the margin (autocorrelated, heavy-tailed
  differences) and power at zero; the delta-method version equals the DM version when the scale is constant; the
  row estimates, P106's contribution and S3 (b) tests, and seed agreement; the verdict labels; the pooled test reduces
  to the single-index and average-series tests; the seed average, the June mask and the top-loss days; the
  claim-level decision rules; and an end-to-end dry run of both stages.

## 14. Checks run before this file was committed (3 November 2026)

* **Dry run.** Both stages ran end to end on three simulated markets (`--synthetic`, SYNTHETIC: 200 days, seed sets
  7 and 101, 15 and 30 minutes): 12 jobs of 7 variants in about 1.3 minutes, and `analyze` wrote every file listed in
  Section 8, with 264 rows in `tests.csv`.
* **Tests.** `PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_p110_gain_stress_test.py`: 12 passed
  (SYNTHETIC).
* **Not run on real data:** nothing. The committed settings of the Bank Nifty and Fin Nifty runs were read from their
  `run_info.json` and match P106's `PIPELINE_SETTINGS` (a test checks this).
