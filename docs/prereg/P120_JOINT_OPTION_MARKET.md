# P120 pre-registration: is the hybrid's gain free of option-market information when the option-implied and VIX inputs are removed together?

**Status.** Written on 3 November 2026 (phase 1) by an AI agent (Claude) working for Shreyan Thanki, at the request
of the project's lead agent, in answer to Major comment 4 of the second AI referee report. **This design is committed
to git and pushed before any outcome of this study has been computed.** No forecast of this study has been made on
real data, and no loss, contribution, interval, test statistic or bootstrap draw of this study has been computed on
real data. The only real-data numbers used below are committed results of earlier studies (P106's
`results/p106_gain_decomposition/` and P110's `results/p110_gain_stress_test/`) and simple arithmetic on them
(Section 7 says exactly which). The commit that adds this file is the time stamp. The code that phase 2 runs
(`experiments/p120_joint_option_market.py`, which imports P106's and P110's code) and its tests
(`tests/test_p120_joint_option_market.py`, synthetic data only) will be committed and pushed in a separate commit
**after** this one and **before** the first real-data forecast; that code must refuse to run on real data unless this
file is in the git history. Phase 2 writes `results/p120_joint_option_market/` and
`docs/P120_JOINT_OPTION_MARKET.md`. All results will be REAL DATA.

## In plain words

Our hybrid forecasts how much an index will move over the next 15 or 30 minutes. It starts from the anchor paper's
model (mcsGARCH, in our implementation) and multiplies its forecast by a correction that LightGBM learns from 41
inputs. Two of the input groups come from the option market: the **option-implied group** (16 inputs computed from
the previous evening's option chain) and **India VIX** (1 input). The project's documents say the hybrid's gain comes
from recent realized volatility and the time of day, **not from option-market information**.

That sentence rests on studies that removed **one group at a time** (P106 on Nifty, P110 on Bank Nifty and Fin
Nifty). Removing one group at a time can hide a joint effect: if the option chain and VIX carry the same information,
the hybrid can lose either one without harm, because the other stands in for it. P106 did remove both groups
together (the `optionmkt` variant), but only on Nifty, and it only asked "does removing them make the hybrid
worse?" (it did not find that: +0.60 points, p 0.16, at 15 minutes). Failing to find an effect is not evidence that
there is none. The second AI referee report therefore asks for an **equivalence test** of the joint removal before the
documents may say the gain does not come from option-market information.

This study does exactly that:

* It refits the hybrid without the option-implied and VIX groups **together** on Bank Nifty and Fin Nifty (never
  done before), with the same five seed sets and the same walk-forward as P110, changing nothing else.
* On Nifty it uses P106's existing `optionmkt` forecasts (all five seed sets were run in P106).
* It asks, for each index and horizon, whether the data can rule out that the two groups together are worth 1 point
  (1% of mcsGARCH's average loss) or more, in either direction.

## Glossary

* **QLIKE:** the loss used to score a variance forecast f of a realized variance y: y/f - log(y/f) - 1. Lower is
  better. It is averaged within each test day first, then over days.
* **mcsGARCH:** the anchor paper's intraday volatility model, in our implementation (an expiry-day time-of-day
  curve, 5-minute smoothing, zero mean; P116 describes how it differs from the paper's specification).
* **The hybrid (`full`):** mcsGARCH's forecast times exp(a LightGBM correction) learnt from 41 inputs; the headline
  forecaster (`hybrid_base`).
* **Point:** 1% of mcsGARCH's mean daily QLIKE on that index and horizon. A difference between two models' mean
  QLIKE, divided by mcsGARCH's mean QLIKE and multiplied by 100, is in points. The hybrid's gain over mcsGARCH, in
  points, is the same number as its margin in %.
* **`optionmkt`:** the hybrid refitted without the 17 option-market inputs (the 16 option-implied ones and India
  VIX), keeping the other 24. Defined in P106's `DROP_SETS`.
* **J (the joint contribution):** 100 x (mean QLIKE of `optionmkt` - mean QLIKE of `full`) / mean QLIKE of mcsGARCH.
  Positive J means removing the option-market inputs makes the hybrid worse, so the option market helps.
* **Seed set:** a base seed s that fixes LightGBM's randomness (five LightGBM seeds s, s + 1000, ..., s + 4000,
  averaged, and the pass that picks the number of boosting rounds). Seed set 7 is the headline; 101 to 104 are four
  more, as in P99, P106 and P110.
* **DM test:** the project's Diebold-Mariano test of a daily loss difference (Newey-West variance, the
  Harvey-Leybourne-Newbold small-sample factor, Student t with n - 1 degrees of freedom).
* **Equivalence test (TOST, two one-sided tests):** with a margin delta, it tests whether the true J lies inside
  (-delta, +delta). It passes (p below 0.05) exactly when the 90% confidence interval of J lies inside the margin.
* **Holm adjustment:** a correction of p-values for testing several rows at once; it keeps the chance of any false
  "pass" in a family at 5%.
* **Driscoll-Kraay errors:** standard errors for an average over several indices that allow the indices to move
  together on the same day and over neighbouring days (P107's method).

## Why this study

The second AI referee report (round 2), Major comment 4: the claim "the gain comes from realized-volatility history and
time of day, not from option-market inputs" is stated more firmly than the ablation allows. P106 removed one input
group at a time; removing the option-implied and VIX groups together raised the 15-minute loss by 0.60 points on
Nifty (`results/p106_gain_decomposition/tests.csv`, row `S3_next15_optionmkt`, p 0.160), and that joint row was never
tested for equivalence. P110 ran equivalence tests per group on Bank Nifty and Fin Nifty, but never the joint drop
set. This study supplies the missing joint equivalence test on all three indices.

## Prior knowledge (what was known when this was written)

All REAL DATA, from committed files. "Points" as in the glossary.

**Nifty: the `optionmkt` rows are not blind.** P106 ran `optionmkt` on Nifty with all five seed sets at 15 and 30
minutes and with seed set 7 for the last hour. What has already been published:

* `results/p106_gain_decomposition/tests.csv`, seed set 7 only (one-sided DM, "removing raises the loss"):
  * `S3_next15_optionmkt`: J = +0.598 points, DM statistic 0.995, p 0.160 (Holm 0.641 over 6 rows);
  * `S3_next30_optionmkt`: J = -0.062, DM statistic -0.153, p 0.561 (Holm 1.000);
  * `E4_last_hour_optionmkt`: J = +0.363, DM statistic 0.421, p 0.337 (exploratory, no adjustment);
  * `S3b_next15_optionmkt` and `S3b_next30_optionmkt`: the period changes (before and after 20 November 2024).
* `results/p106_gain_decomposition/seed_repeats.csv` (rows "1 (all days)", variant `optionmkt`): J in **each of the
  five seed sets** (7, 101, 102, 103, 104):
  * 15 minutes: +0.598, +0.155, +0.488, +0.184, +0.374 (mean +0.360, SD 0.191, all five positive);
  * 30 minutes: -0.062, +0.045, +0.356, +0.060, -0.121 (mean +0.056, SD 0.184, three of five positive).
* `docs/P106_GAIN_DECOMPOSITION.md` (Section 3) states the 15-minute numbers in words: "+0.60 points at 15 minutes
  (positive in all five seed sets, mean +0.36) and -0.06 at 30 minutes, neither significant".
* The committed daily losses `results/p106_gain_decomposition/daily_losses_nifty_<next15|next30>_s<7|101|102|103|104>.csv`
  and `daily_losses_nifty_last_hour_s7.csv` contain an `optionmkt` column.

Because mcsGARCH's loss is the same in every seed set, the estimate J on the five-seed average equals the mean of the
five seed-set estimates. So **the Nifty point estimates of this study's primary rows are already public: +0.360
points at 15 minutes and +0.056 at 30 minutes.** What has never been computed is their equivalence test: the standard
error of the five-seed average, its 90% interval and the TOST p-value. The Nifty rows are therefore a re-analysis of
public data with a new test, not new evidence. They will be computed from P106's committed daily losses (Section 2),
checked against P106's git-ignored work files, and one Nifty `optionmkt` job will be rerun to check that the code path
reproduces P106's forecasts exactly.

**Bank Nifty and Fin Nifty: `optionmkt` has never been run** (P106 ran only the five single groups there with seed
set 7; P110 ran the five single groups and `mcsinfo` with seed sets 7 and 101 to 104, but not `optionmkt`; P110's
plan, Section 1, says so). What is known about related variants on these two indices (REAL DATA, P110):

* **The two groups one at a time** (`results/p110_gain_stress_test/equivalence_bounds.csv` and `verdict.csv`,
  average over seed sets 101 to 104, TOST at 1 point, Holm over P110's 12 F2 rows):

  | Index, horizon | option-implied: estimate [90% interval], Holm p | VIX: estimate [90% interval], Holm p |
  |---|---|---|
  | Bank Nifty, 15 min | -0.22 [-0.67, +0.24], 0.007 | +0.08 [-0.01, +0.17], below 0.001 |
  | Bank Nifty, 30 min | -0.20 [-0.82, +0.42], 0.035 | +0.14 [+0.01, +0.27], below 0.001 |
  | Fin Nifty, 15 min | +0.09 [-0.35, +0.52], 0.001 | +0.05 [-0.04, +0.13], below 0.001 |
  | Fin Nifty, 30 min | -0.07 [-0.83, +0.69], 0.035 | -0.12 [-0.25, +0.02], below 0.001 |

  All eight rows were "equivalent" within 1 point; the two option-implied rows at 30 minutes failed under P110's
  stricter readings (one Holm over all 28 primary rows, Holm 0.088; and in most robustness samples). With all five
  seed sets averaged (P110 rows `R2_F2_*`), the option-implied 90% intervals were [-0.65, +0.22] and [-0.88, +0.36]
  on Bank Nifty and [-0.33, +0.55] and [-0.77, +0.72] on Fin Nifty (`tests_detail.csv`).
* **The `mcsinfo` hybrid**, which removes the option-implied group, VIX **and** the calendar group except the expiry
  flag (24 inputs), against the full hybrid (`verdict.csv`, `equivalence_bounds.csv`; positive = the full hybrid is
  better; non-inferiority at 1 point, Holm over P110's 8 F3 rows): Bank Nifty +0.13 (upper end of the 90% interval
  0.57, "non-inferior", Holm 0.002) and +0.23 (0.83, "not shown", Holm 0.055); Fin Nifty +0.65 (1.17, "not shown",
  Holm 0.272) and +0.50 (1.33, "not shown", Holm 0.272), with every seed set putting `mcsinfo` behind the full hybrid
  on Fin Nifty (by 0.56 to 0.76 points at 15 minutes and 0.42 to 0.68 at 30). So on Fin Nifty the information
  mcsGARCH lacks, taken together, may be worth about half a point, and more than 1 point was not ruled out. `mcsinfo`
  removes more than `optionmkt` does (the calendar inputs too), so this bounds nothing about J directly, but it is the
  closest existing number.
* **Nifty's single groups** (P110 `S5_*` rows, average over seed sets 101 to 104): option-implied +0.12 [-0.50,
  +0.74] and -0.08 [-0.64, +0.48]; VIX +0.03 [-0.08, +0.14] and -0.09 [-0.20, +0.03]; all "equivalent".
* **Last hour:** the only `optionmkt` last-hour run is Nifty's seed set 7 (above). The full hybrid's last hour exists
  with seed sets 101 to 104 only on Nifty (P99, `results/seed_bootstrap/daily_losses_last_hour.csv`); on Bank Nifty
  and Fin Nifty only the committed seed-7 pipelines have a last-hour hybrid (`results/pipeline_nse_banknifty/`,
  `results/pipeline_nse_finnifty/`).

## 1. Variants and seed sets

**Variants** (P106's definitions, unchanged; `GROUPS` and `DROP_SETS` in `experiments/p106_gain_decomposition.py`):

| Variant | Inputs removed from the 41 | Model name |
|---|---|---|
| `full` | none | `hybrid_base` |
| `optionmkt` | the 16 option-implied inputs (`implied` group) and `log_vix` (`vix` group): 17 inputs | `hybrid_drop_optionmkt` |

**Seed sets:** 7, 101, 102, 103 and 104, exactly as in P106 and P110.

## 2. Design

**What is fitted.** P106's walk-forward (`run_walkforward` with the committed `WalkForwardConfig`: 150 training days,
refit before every block of 21 trading days, mcsGARCH refitted every 21 days, LightGBM default hyperparameters with the
number of rounds chosen inside each training window, five-seed averaging), through P106's own job function
(`run_job`, `make_config`) and settings check (the run refuses to start if any setting, the number of days or the last
day differs from the committed run of that index). Nothing is retuned.

| Index | Target | Seed sets | Variants fitted here | Jobs | What is reused instead of refitted |
|---|---|---|---|---|---|
| Bank Nifty | next 15, next 30 | 7, 101 to 104 | `optionmkt` | 10 | `full`: P110's forecasts (work files `data/processed/p110/`, git-ignored) and committed daily losses (`results/p110_gain_stress_test/daily_losses_banknifty_<target>_s<seed>.csv`) |
| Fin Nifty | next 15, next 30 | 7, 101 to 104 | `optionmkt` | 10 | the same, Fin Nifty |
| Bank Nifty | last hour | 7, 101 to 104 | `full`, `optionmkt` | 5 | nothing (never run with these seed sets) |
| Fin Nifty | last hour | 7, 101 to 104 | `full`, `optionmkt` | 5 | nothing |
| Nifty | next 15 | 7 | `optionmkt` | 1 | none: a reproduction check only (compared with P106's work file; not used in any test) |
| Nifty | last hour | 101 to 104 | `full`, `optionmkt` | 4 | seed set 7: P106's committed `daily_losses_nifty_last_hour_s7.csv` |

**Nifty at 15 and 30 minutes is not refitted.** Its daily losses are P106's committed files
`results/p106_gain_decomposition/daily_losses_nifty_<target>_s<seed>.csv` (columns `mcsgarch`, `full`, `optionmkt`,
`implied`, `vix` and the others). The one rerun job above checks that P120's code path makes the same forecasts.

**Test days** (the committed runs' test days; unchanged): Nifty 792 (10 August 2023 to 2 November 2026), Bank Nifty
660 (10 August 2023 to 22 April 2026), Fin Nifty 604 (10 August 2023 to 27 January 2026). The last-hour target scores
one window per day (14:30 to 15:30), so it has the same number of days.

**Primary sample: each model's daily QLIKE averaged over all five seed sets (7, 101, 102, 103, 104).** mcsGARCH's
daily QLIKE must be identical in every seed set (checked; the script refuses otherwise). Why five and not P110's four:
P110 left seed set 7 out of its primary sample because P106 had already published seed-7 results for those variants on
those indices. Here no `optionmkt` result of any seed set exists on Bank Nifty or Fin Nifty, so all five are blind
there, and on Nifty all five are equally public. P110's four-seed sample is robustness R2.

## 3. Statistics

All losses are QLIKE averaged within each day first (P106's `day_frame`). L_x is the mean over days of model x's
daily QLIKE (on the five-seed average for the hybrids).

* **The joint contribution:** J = 100 x (L_optionmkt - L_full) / L_mcs, in points. Positive: the option market helps.
* **The daily series** d_t = 100 x (L_optionmkt,t - L_full,t) / L_mcs (P110's `points_series`), whose mean is J.
* **TOST with margin delta** (P110's `tost`, unchanged): test 1, H0 J >= delta against J < delta (the DM test of
  d_t - delta, alternative "less"); test 2, H0 J <= -delta against J > -delta (the DM test of d_t + delta,
  alternative "greater"); the TOST p is the larger of the two. It is below 0.05 exactly when the 90% interval of J
  lies inside (-delta, +delta). Every row reports the 90% interval and "the smallest margin that would pass" (the
  larger end of the interval in absolute value).
* **"At least 1 point worse without them":** H0 J <= 1 against J > 1: the t statistic (mean(d) - 1) / SE x HLN
  factor, with the DM variance and Student t(n - 1) (the same pieces as P110's `dm_parts`). It is below 0.05 exactly
  when the 90% interval of J lies wholly above +1. Its mirror, H0 J >= -1 against J < -1 ("at least 1 point better
  without them"), is computed for completeness in the same way.
* **Two-sided difference:** H0 J = 0, the project's DM test of L_optionmkt,t against L_full,t, two-sided.
* **Pooled over the three indices** (P110's `pooled_test` with P107's `index_means_dk`): each index's daily series d_t
  in points of its own mcsGARCH QLIKE, on the 604 test days common to the three indices (10 August 2023 to 27 January
  2026); their equal-weight average, Driscoll-Kraay covariance, HLN factor, Student t(T - 1); TOST at 1 point.
* **Delta-method version** (robustness; P110's `tost_delta_method`): the same TOST on the ratio J with the delta-method
  Newey-West variance (P106's `ratio_var`), so that L_mcs is not treated as known.
* **Margins.** delta = **1 point** for the primary rows, the same as P110 and for the same reasons (P110's plan,
  Section 3): P106 named "more than 1 point" as the size that would contradict "adds nothing"; it is far above seed
  noise (about 0.1 to 0.2 points); it is 13% to 26% of the hybrid's gain on the three indices (gains of 3.9 to 7.9
  points). Its weakness, as in P110: Bank Nifty's time of day, one of the two groups said to carry the gain, is worth
  about 0.9 to 1.3 points, so "within 1 point" is not "negligible". So every row is repeated with **0.5 point** (S1),
  and every row reports the smallest passing margin.

**Seed agreement** (as P110). J is also computed on each seed set alone. For an equivalence row a seed set "agrees" if
its J lies inside (-delta, +delta); for a "worse by at least 1 point" row if its J is above +1; for the two-sided
difference if its J has the sign of the five-seed estimate. A row's estimate smaller in absolute value than twice the
SD of the five seed-set estimates is flagged "within seed noise".

## 4. Hypotheses, families and tests

**Primary family H1 (6 tests), Holm over the 6 TOST p-values.** For each index (Nifty, Bank Nifty, Fin Nifty) and
horizon (15 and 30 minutes): H1: -1 < J < 1, the hybrid without the option-implied and VIX groups together has mean
QLIKE within 1 point of mcsGARCH's QLIKE of the full hybrid. TOST, margin 1 point, five-seed average.

**Companion family H1w (6 tests), Holm over the 6 p-values.** For the same six cells: H1: J > 1 (the hybrid is worse
by at least 1 point without them), one-sided, as in Section 3. This family exists only to give the third verdict below;
it cannot make a row "equivalent".

**Verdict of each primary row** (fixed now; the wording mirrors P110's):

| Verdict | Rule |
|---|---|
| **equivalent within 1 point** | H1 TOST Holm p below 0.05 **and** all five seed sets have -1 < J < 1 |
| **worse by at least 1 point** | H1w Holm p below 0.05 **and** all five seed sets have J > 1 |
| **not shown either way** | anything else. Sub-label "seed-fragile" if a Holm p passed but the seed rule failed. |

The two first verdicts cannot both hold (an interval cannot lie inside (-1, 1) and above 1). For completeness: if the
90% interval lay wholly below -1 with every seed set below -1 (the hybrid **better** by at least 1 point without the
option market; Holm over the 6 mirror p-values), the row would be labelled "better by at least 1 point without them".
We do not expect this anywhere.

**What the documents may then say** (fixed now):

* "Removing the option-implied and VIX inputs together changes the loss by less than 1% of mcsGARCH's" may be said
  for an index and horizon only if its row is "equivalent within 1 point".
* "The gain does not come from option-market information" (without qualification, on all three indices) may be said
  only if all six rows are "equivalent within 1 point".
* If some rows are "not shown either way" and none is "worse by at least 1 point", the documents must name the rows
  where more than 1 point is ruled out and the rows where it is not, and may not use the unqualified sentence. P106's
  wording "removing VIX together with the chain does not reveal a hidden option-market contribution" must then be
  replaced by "no significant joint contribution was found, and a joint contribution of 1 point or more is ruled out
  only on [rows]".
* If any row is "worse by at least 1 point", the claim is contradicted for that index and horizon and the documents
  must say that the option market carries at least 1 point of the gain there.

**Secondary families** (each with its own Holm adjustment; they do not change a primary verdict):

* **S1 (6 tests), margin 0.5 point:** the six H1 cells, TOST with delta = 0.5. Same verdict rule with 0.5 in place of
  1 ("equivalent within 0.5 point" / "not shown"; seed rule -0.5 < J < 0.5).
* **S2 (6 tests), is J different from zero at all?** The six cells, two-sided DM test of L_optionmkt against L_full.
  "Different from zero" if Holm p below 0.05 and all five seed sets share the sign of the estimate; otherwise "no
  difference shown".
* **S3a (3 tests), last hour, TOST at 1 point:** Nifty, Bank Nifty and Fin Nifty, the last-hour target (one 60-minute
  window per day, from 14:30), five-seed average; the H1 verdict rule. Holm over 3.
* **S3b (3 tests), last hour, two-sided DM:** as S2, for the last hour. Holm over 3.
* **S4 (2 tests), pooled three-index TOST at 1 point:** at 15 and at 30 minutes, the equal-weight average over Nifty,
  Bank Nifty and Fin Nifty on their 604 common test days, Driscoll-Kraay errors (P107's approach, reused through
  P110's `pooled_test`; the per-index seed rule is not applied to a pooled row). Holm over 2. Label "equivalent within
  1 point (pooled)" or "not shown (pooled)".

**Robustness** (each family the six H1 TOST rows with its own Holm over 6; reported next to the primary verdicts,
never replacing them):

* **R1:** seed set 7 alone (for Nifty this re-derives P106's public seed-7 number with the new test).
* **R2:** the average over the four new seed sets 101 to 104 (P110's primary sample).
* **R3:** the delta-method variance of the ratio.
* **R4:** without 4 and 5 June 2024 (the general-election result day and the day after).
* **R5:** without, on each index and target, the 1% of test days with the largest mcsGARCH daily QLIKE (7, 6 and 6
  days); the rule uses the benchmark's loss only.

**Exploratory** (no adjustment; descriptive in weight):

* **E1 (6 tests), do the chain and VIX stand in for each other?** The interaction I = J - C_implied - C_vix, where
  C_implied and C_vix are the single-group contributions on the same five-seed average (Bank Nifty and Fin Nifty from
  P110's committed daily losses, Nifty from P106's). Daily series 100 x (L_optionmkt,t - L_implied,t - L_vix,t +
  L_full,t) / L_mcs, two-sided DM. Positive I: removing both costs more than the sum of removing each, the sign of
  substitutes.
* **E2 (6 tests), expiry days only:** the H1 TOST at 1 point on the index's expiry days (P106's `is_expiry`), five-seed
  average.

**Descriptive (no test):** J in every seed set; J as a share of the hybrid's gain over mcsGARCH with a 95% stationary
bootstrap interval (P106's `share_ci`, 10,000 draws, mean block 10 days); J next to C_implied, C_vix and the `mcsinfo`
gap N of P110 and P106 on the same sample.

Every p-value goes in `results/p120_joint_option_market/tests.csv` with P110's columns (test_id, hypothesis, role,
sample, estimate, statistic, p_raw, sidedness, p_method, p_adjusted, adjust_method). For TOST rows the statistic is
that of the binding (larger-p) side. Any test added after this commit is labelled "post hoc".

## 5. The seed rule and the label "within seed noise"

As P110 (Section 3 above). A primary row needs all five seed sets to agree. With five seed sets this is strict: a
single seed set outside (-1, +1) makes an otherwise passing row "not shown either way (seed-fragile)".

## 6. What we expect (stated in advance; it changes no rule)

| Row | Expectation | Reason |
|---|---|---|
| H1 Nifty, 15 min | **not shown either way** (likely) | the public five-seed estimate is +0.360; seed set 7's own 90% interval reached +1.59 (Section 7); unless the five-seed average is much more precise (SE below about 0.39 points), the interval will reach past +1 |
| H1 Nifty, 30 min | equivalent within 1 point | the public estimate is +0.056 and seed set 7's interval was [-0.73, +0.60] |
| H1 Bank Nifty and Fin Nifty, 15 min | equivalent within 1 point | the option-implied group alone had intervals ending 0.52 to 0.67 points from zero and VIX adds little |
| H1 Bank Nifty and Fin Nifty, 30 min | uncertain: equivalent or not shown | the option-implied group alone reached 0.82 and 0.83 points (0.88 and 0.77 with five seed sets); a joint removal may be larger, and the Holm step-down makes the last rows hardest |
| H1w | no row "worse by at least 1 point" | no seed-averaged estimate of a single group or of `mcsinfo`'s gap N is above +0.7 points (P106, P110) |
| S1 (0.5 point) | not shown on most rows | the option-implied group alone never passed at 0.5 point in P110 |
| S2 | no row different from zero, except perhaps Nifty at 15 minutes | Nifty 15-minute seed estimates were all positive (+0.16 to +0.60); seed set 7 alone had p 0.160 one-sided |
| S3 last hour | not shown | the last hour has one window per day; seed set 7's Nifty interval was [-1.06, +1.78] |
| S4 pooled | equivalent at both horizons | pooling narrowed P110's option-implied intervals to about +/- 0.3 to 0.5 points |
| E1 | positive on Nifty at 15 minutes | there J (+0.36) is larger than C_implied + C_vix (+0.12 - 0.01 = +0.11, P106 `seed_repeats.csv`) |

## 7. Power, from public numbers (computed before this file was committed)

From P106's committed `tests.csv` only, as P110's plan did: SE = estimate x sqrt((n - 1)/n) / DM statistic, the 90%
interval estimate +/- t(0.95, n - 1) x SE / sqrt((n - 1)/n). REAL DATA, Nifty, seed set 7, 792 days, points:

| Row | J | SE | 90% interval | smallest passing margin |
|---|---|---|---|---|
| `S3_next15_optionmkt` | +0.60 | 0.60 | [-0.39, +1.59] | 1.59 |
| `S3_next30_optionmkt` | -0.06 | 0.40 | [-0.73, +0.60] | 0.73 |
| `E4_last_hour_optionmkt` | +0.36 | 0.86 | [-1.06, +1.78] | 1.78 |

For scale on the other two indices: P110's five-seed standard errors of the option-implied group alone were 0.26 and
0.38 points (Bank Nifty, 15 and 30 minutes) and 0.27 and 0.45 (Fin Nifty); of VIX alone 0.05 to 0.08
(`results/p110_gain_stress_test/tests_detail.csv`, rows `R2_F2_*`). On Nifty, averaging five seed sets lowered the
option-implied group's SE only from 0.46 to 0.39 (P110's plan, Section 7): the width comes mostly from which days
happened, not from the seeds. With 6 rows in H1, Holm compares the smallest TOST p with 0.05/6 = 0.0083 and the largest
with 0.05. For the Nifty 15-minute row to pass, its five-seed SE would have to be below about (1 - 0.36) / 1.647 = 0.39
points; seed set 7 alone gave 0.60. **The Nifty 15-minute row is the one most likely to stay "not shown"; the plan
accepts this and the write-up will report it plainly.** No standard error of J can be computed in advance for Bank
Nifty or Fin Nifty.

## 8. The exact commands

From the repository root, at the code commit that follows this file, with the raw data listed in `data/MANIFEST.csv`
in `data/raw`, P110's work files in `data/processed/p110/` and P106's in `data/processed/p106/` (both git-ignored; the
analysis falls back to the committed daily losses if they are missing, and reports the checks that need them as "not
run"):

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py run --index BANKNIFTY --workers 2 \
    > results/p120_joint_option_market/run_stdout_banknifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py run --index FINNIFTY --workers 2 \
    > results/p120_joint_option_market/run_stdout_finnifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py run --index NIFTY --workers 2 \
    > results/p120_joint_option_market/run_stdout_nifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py analyze --plan-commit <this commit> \
    > results/p120_joint_option_market/analyze_stdout.txt 2>&1
```

`run` uses at most 2 worker processes, stores each variant's forecasts in `data/processed/p120/` (git-ignored; a
finished job is not rerun) and the SHA-256 of every raw input. Within an index the jobs run in this order: the
15- and 30-minute `optionmkt` jobs (seed set 7, then 101 to 104), then the last-hour jobs. `analyze` writes
`tests.csv`, `tests_detail.csv`, `verdict.csv`, `equivalence_bounds.csv`, `contributions.csv`, `seed_repeats.csv`,
`shares.csv`, `pooled.csv`, `reproduction_check.csv`, `daily_losses_<index>_<target>_s<seed>.csv` (the `optionmkt` and
`full` daily QLIKE of every new run, from which every number can be recomputed), `timing.csv` and `run_info.json`.
Every results file is labelled REAL DATA.

## 9. Checks inside the run

* `y`, `mcsgarch` and `mcsgarch_iv` in every new `optionmkt` work file must equal, row by row, those of P110's `full`
  work file of the same index, target and seed set (when those git-ignored files are on disk); the daily QLIKE of
  mcsGARCH must equal P110's committed daily losses on every day, and the `full` hybrid's daily QLIKE recomputed from
  P110's work files must equal P110's committed `full` column.
* The rerun Nifty `optionmkt` job (15 minutes, seed set 7) must equal P106's work file forecast by forecast; the Nifty
  daily losses recomputed from P106's work files (`full`, `optionmkt`, every seed set, 15 and 30 minutes) must equal
  P106's committed daily losses.
* The new last-hour `full` runs: seed set 7 on Bank Nifty and Fin Nifty must reproduce the committed `hybrid_base`
  last-hour forecasts of `results/pipeline_nse_banknifty/` and `results/pipeline_nse_finnifty/`; seed sets 101 to 104
  on Nifty must reproduce P99's daily losses (`results/seed_bootstrap/daily_losses_last_hour.csv`).
* mcsGARCH's daily QLIKE must be identical in all five seed sets of an index and target (the script refuses
  otherwise).
* Every raw file read must agree with `data/MANIFEST.csv` (recorded in `run_info.json`).

A failed check is reported next to the results it concerns, with its cause if found; the results are still given.

## 10. What will be reported, whatever the result, and the deviations policy

In `docs/P120_JOINT_OPTION_MARKET.md`, for a beginner reader, every number labelled REAL DATA and taken from a file in
`results/p120_joint_option_market/` (or, where stated, a committed file of P106 or P110): the answer first; the six
H1 rows with estimate, 90% interval, TOST Holm p, the H1w Holm p, seed agreement and verdict; S1 to S4; robustness,
exploratory and descriptive results with their labels; the reproduction checks; the exact commands and run time; a
"Deviations from the plan" section; limitations; and the sentence the project's documents may use under Section 4's
rules.

* After this file is committed the design does not change. Any change is a deviation, listed with its reason and with
  whether any outcome had been seen when it was decided.
* A crash is fixed with the smallest change that lets the run finish, and only if it changes no number of a job that
  already finished.
* **Stopping rule.** The run stops when every job in Section 2 has finished. If time runs short (the lead agent's
  deadline is 18:30 UTC on 3 November 2026), the jobs are cut from the end: first the Nifty last-hour jobs, then Fin
  Nifty's and Bank Nifty's last-hour jobs; S3 rows without all five seed sets of every index they need are "not
  computed", with the reason. H1 rows are never computed on fewer than five seed sets; if a 15- or 30-minute job
  cannot be run, its rows are "not computed".
* Nothing is rerun to get a different answer; no seed set is added or dropped after results are seen.

## 11. Expected run time

From P106's and P110's timings (`results/p110_gain_stress_test/timing.csv`, `results/p106_gain_decomposition/timing.csv`):
a variant fit on Bank Nifty or Fin Nifty takes about 55 to 85 seconds, and loading a market and building its datasets
about 2.5 to 3 minutes. Bank Nifty: 10 one-variant jobs and 5 two-variant jobs, about 20 to 30 fit-minutes; the same
for Fin Nifty; Nifty: 1 one-variant job and 4 two-variant jobs, about 10 to 15 fit-minutes. With 2 worker processes,
about 45 to 75 minutes of wall-clock time in all. The analysis takes about 1 minute.

## 12. Known weaknesses

1. **No new days.** All tests use the same test days as P106 and P110; Nifty's point estimates are public (Prior
   knowledge). New seed sets and a new test remove one source of chance, not the other (which days happened).
2. **The margin is a choice.** 1 point is about a sixth to a quarter of the gain on these indices and about the size
   of Bank Nifty's time-of-day contribution. The 0.5-point rows and the reported intervals let a reader use another
   margin.
3. **Removing a block of inputs measures what it adds given the others.** The 24 remaining inputs include mcsGARCH's
   own forecast and the realized history, which may already carry some of what the option market knows (for example
   recent volatility moves VIX). "The hybrid does not need the option market" is what can be tested; "the option market
   holds no information about the next 15 minutes" is not.
4. **The protocol is fixed.** The `optionmkt` hybrid keeps the headline's LightGBM settings; tuned, it might do
   better or worse.
5. **Five seed sets** show whether a sign is stable; the DM and TOST p-values do not include seed noise.
6. **The three indices move together** (P107: about two independent indices' worth), so three agreeing rows are less
   than three independent confirmations, and the pooled row averages correlated evidence.
7. **The scale of the points is treated as known** in the main tests (R3 relaxes it).
8. **The last hour** has one forecast window per day and is noisy; S3 is secondary for that reason.

## 13. Code

* `experiments/p120_joint_option_market.py`, new: `run` (forecasts per index with P106's `make_config`, `run_job`,
  work files and settings check; at most 2 worker processes) and `analyze` (Sections 3 to 5, using P110's `dm_parts`,
  `tost`, `tost_delta_method`, `points_series`, `pooled_test`, `date_mask`, `top_loss_days` and P106's `day_frame`,
  `share_ci`, `holm`; a seed average over all variant columns, because P110's `seed_average` averages only P110's seven
  variants). It refuses to run on real data unless this plan is in the git history, and `analyze` refuses a
  `--plan-commit` that is not an ancestor of the checked-out commit or does not contain this file. `--synthetic` runs
  both stages on simulated markets (labelled SYNTHETIC).
* No change to `src/`, to P106's, P107's or P110's files, or to their behaviour.
* `tests/test_p120_joint_option_market.py`, new, synthetic data only: the drop set, the job plan, the seed average,
  the "worse by at least 1 point" test and its mirror against shifted DM tests and the 90% interval rule, the verdict
  rules, the plan-in-history check, and an end-to-end dry run.

## 14. Checks run before this file was committed (3 November 2026)

* The arithmetic in Section 7 and the numbers quoted under Prior knowledge were read from the committed files named
  next to them.
* `python -m pytest -q tests/test_reproduce_all.py` passes with `P120_JOINT_OPTION_MARKET` listed in `PENDING` in
  `scripts/reproduce_all.py`.
* Not run on real data: nothing of this study.
