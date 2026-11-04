# P120: is the hybrid's gain free of option-market information when the option-implied and VIX inputs are removed together?

Study P120, run by an AI agent (Claude) working for Shreyan Thanki on 4 November 2026 (the plan was written and
committed on 3 November 2026 and never run that day). **All numbers below are REAL DATA** (Nifty through P106's
committed daily losses, Bank Nifty from new refits on the NSE data of the headline run) unless marked SYNTHETIC.
Every number comes from a file in `results/p120_joint_option_market/` (named next to each table) or, where marked,
from an earlier committed file of P106 or P110.

* Registered plan: `docs/prereg/P120_JOINT_OPTION_MARKET.md`, commit `30da69e` (committed and pushed on 3 November
  2026 before any forecast or loss of this study was computed).
* Code: `experiments/p120_joint_option_market.py`, which reuses P106's walk-forward, work files and settings check and
  P110's tests. Tests (SYNTHETIC only): `tests/test_p120_joint_option_market.py`, 9 passed before the real-data run.
* Every p-value of the study: `results/p120_joint_option_market/tests.csv`.
* This run had 70 minutes of wall clock in all, with the raw NSE data still downloading when it started. Section 7
  lists what that cut from the plan. **Fin Nifty was not run at all**, so its two primary rows, the pooled rows and
  the Nifty and Fin Nifty last-hour rows are NOT COMPUTED; Section 7 says why. All 15 Bank Nifty jobs of the plan
  ran (the 15- and 30-minute `optionmkt` jobs and the last-hour jobs, five seed sets each).

## 1. The answer in plain words

Our hybrid forecasts how much an index will move over the next 15 or 30 minutes. It starts from the anchor paper's
model (mcsGARCH, in our implementation) and multiplies its forecast by a correction that LightGBM learns from 41
inputs. Seventeen of those inputs come from the option market: 16 computed from the previous evening's option chain
(the option-implied group) and India VIX. The project's documents say the hybrid's gain comes from recent realized
volatility and the time of day, not from option-market information. The second AI referee asked for the missing test:
remove the two option-market groups **together** and show, with an equivalence test, that the hybrid loses less than
1 point (1% of mcsGARCH's average loss, QLIKE) in either direction.

**The short answer** (REAL DATA, `verdict.csv`, by the rules fixed in the plan):

| Index, horizon | J (points; positive: the option market helps) | 90% interval | Verdict |
|---|---|---|---|
| Nifty, 15 min | +0.36 | [-0.37, +1.09] | **not shown either way** |
| Nifty, 30 min | +0.06 | [-0.56, +0.67] | **equivalent within 1 point** |
| Bank Nifty, 15 min | -0.18 | [-0.67, +0.32] | **equivalent within 1 point** |
| Bank Nifty, 30 min | -0.20 | [-0.90, +0.49] | **not shown either way** (the raw equivalence test passes, p 0.030; the Holm-adjusted one does not, p 0.060) |
| Fin Nifty, 15 min | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED (no Fin Nifty raw data on this machine; Section 7) |
| Fin Nifty, 30 min | NOT COMPUTED | NOT COMPUTED | NOT COMPUTED (same) |

* On no row is the hybrid **worse by at least 1 point** without the option market (the companion test H1w never
  comes close; every seed set's J lies inside (-1, +1) on every computed primary row).
* On Nifty at 15 minutes the option market's joint value is +0.36 points, but the 90% interval reaches +1.09, so a
  cost of 1 point or more is **not ruled out** there. The plan said in advance (its Section 6 and 7) that this row
  would most likely stay "not shown", because the width of the interval comes from which days happened, not from the
  seeds. It was right.
* On Nifty at 30 minutes the data rule out a joint value of 1 point or more in either direction.
* On Bank Nifty the hybrid is, if anything, slightly **better** without the option market (J = -0.18 and -0.20
  points, every seed set negative at 15 minutes). At 15 minutes the data rule out a joint value of 1 point or more
  in either direction (90% interval [-0.67, +0.32]). At 30 minutes the interval [-0.90, +0.49] also lies inside the
  margin, so the raw equivalence test passes (p 0.030), but after the Holm step-down over the computed rows it does
  not (p 0.060), and the plan's rule makes the row "not shown either way". The plan called this row "uncertain:
  equivalent or not shown" in advance.
* **The last hour on Bank Nifty is the surprise** (secondary family S3, REAL DATA, `verdict.csv`, `shares.csv`): in
  the one 60-minute window per day from 14:30, the hybrid is **better without** the option-market inputs by 2.40
  points (90% interval [-4.09, -0.70]; every seed set between -1.99 and -2.81; two-sided p 0.020, "different from
  zero"). That is 30% of the hybrid's last-hour gain of 8.0 points (bootstrap interval 5% to 74%). By the plan's
  rule the equivalence verdict is "not shown either way" (the interval lies outside the margin) and the mirror test
  "better by at least 1 point without them" has p 0.088, just short of 5%, so no label is attached; but the sign and
  size are consistent across seed sets, and the documents should say so.
* Fin Nifty was not run (no raw data on this machine; Section 7), so its two rows, the Nifty and Fin Nifty last-hour
  rows and the pooled rows are NOT COMPUTED.

**What the documents may now say** (plan Section 4, fixed in advance): the unqualified sentence "the gain does not
come from option-market information" may **not** be used, because not all six rows are "equivalent within 1 point".
The sentence to use instead is in Section 10.

## 2. The primary rows (H1 and H1w)

REAL DATA, `results/p120_joint_option_market/verdict.csv` and `equivalence_bounds.csv`. The sample of each row is the
daily QLIKE averaged over the five seed sets 7, 101, 102, 103 and 104 (the plan's primary sample). Nifty: 792 test
days, 10 August 2023 to 2 November 2026. Bank Nifty: 660 test days, 10 August 2023 to 22 April 2026. J is in points
(1 point = 1% of mcsGARCH's mean daily QLIKE on that index and horizon); the TOST passes when the 90% interval lies
inside (-1, +1). Holm adjusts the p-values over the four computed H1 rows (the plan said six; Section 7).

| Row | J | SE | 90% interval | TOST p (raw) | TOST p (Holm) | H1w p (Holm) | Seed sets with -1 < J < 1 | Seed-set J (min to max) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| H1 Nifty, 15 min | +0.360 | 0.445 | [-0.373, +1.092] | 0.075 | 0.075 | 1.000 | 5 of 5 | +0.155 to +0.598 | not shown either way |
| H1 Nifty, 30 min | +0.056 | 0.376 | [-0.563, +0.675] | 0.006 | 0.018 | 1.000 | 5 of 5 | -0.121 to +0.356 | equivalent within 1 point |
| H1 Bank Nifty, 15 min | -0.177 | 0.301 | [-0.673, +0.318] | 0.003 | 0.013 | 1.000 | 5 of 5 | -0.227 to -0.086 | equivalent within 1 point |
| H1 Bank Nifty, 30 min | -0.204 | 0.423 | [-0.901, +0.493] | 0.030 | 0.060 | 1.000 | 5 of 5 | -0.403 to +0.018 | not shown either way |
| H1 Fin Nifty, 15 min | NOT COMPUTED | | | | | | | | NOT COMPUTED |
| H1 Fin Nifty, 30 min | NOT COMPUTED | | | | | | | | NOT COMPUTED |

The smallest margin that would pass on each row (`equivalence_bounds.csv`, column `smallest_margin_that_passes`, with
the hybrid's gain over mcsGARCH on the same sample): Nifty 15 min 1.09 points (the hybrid's gain is 5.79 points, so
the bound is 19% of the gain); Nifty 30 min 0.67 points (gain 7.84 points, 9%); Bank Nifty 15 min 0.67 points (gain
4.09 points, 16%); Bank Nifty 30 min 0.90 points (gain 5.26 points, 17%).

Both Nifty estimates and the Bank Nifty 30-minute estimate are "within seed noise" (smaller in absolute value than
twice the SD of the five seed-set estimates: 0.19, 0.18 and 0.16 points): the sign of J is not settled there. The
Bank Nifty 15-minute estimate (-0.18, seed SD 0.06) is not within seed noise: all five seed sets put the hybrid a
little ahead without the option market, by 0.09 to 0.23 points, although the two-sided test (S2 below) does not call
even that a significant difference.

The Nifty rows are a re-analysis of public numbers with a new test: their point estimates (+0.360 and +0.056) were
already in `results/p106_gain_decomposition/seed_repeats.csv` and the plan's "Prior knowledge" section. What is new is
the standard error, the interval and the TOST p-value.

## 3. Secondary families

**S1, margin 0.5 point** (REAL DATA, `verdict.csv`, rows `S1_*`; Holm over the four computed S1 rows). Nifty 15 min:
TOST raw p 0.376 (Holm 0.484), not shown; Nifty 30 min: 0.119 (0.475), not shown; Bank Nifty 15 min: 0.142 (0.475),
not shown; Bank Nifty 30 min: 0.242 (0.484), not shown. As the plan expected, no row shows equivalence within half a
point (the intervals are 1.2 to 1.5 points wide).

**S2, is J different from zero at all?** (two-sided DM, `verdict.csv`, rows `S2_*`). Nifty 15 min: raw p 0.419 (Holm
1.000), seed sets with the sign of the estimate 5 of 5, "no difference shown"; Nifty 30 min: 0.882 (1.000), 3 of 5,
"no difference shown"; Bank Nifty 15 min: 0.555 (1.000), 5 of 5, "no difference shown"; Bank Nifty 30 min: 0.630
(1.000), 4 of 5, "no difference shown". Removing the two groups together does not change the loss by a detectable
amount on any computed row.

**S3, the last hour** (REAL DATA, `verdict.csv`, `equivalence_bounds.csv`, `contributions.csv`, `shares.csv`; the
last-hour target scores one 60-minute window per day from 14:30; five-seed average; Holm over the computed rows,
here one per family).

| Row | J | SE | 90% interval | S3a TOST p | mirror "better by at least 1 point" p | S3b two-sided p | Seed-set J | Verdicts |
|---|---|---|---|---|---|---|---|---|
| Bank Nifty, last hour (660 days) | -2.397 | 1.029 | [-4.093, -0.701] | 0.912 | 0.088 | 0.020 | -2.473, -1.986, -2.029, -2.687, -2.810 (sets 7, 101 to 104) | S3a: not shown either way; S3b: different from zero (5 of 5 seed sets negative) |
| Nifty, last hour | NOT COMPUTED | | | | | | | the four new seed-set jobs were not run (Section 7); P106's public seed-7 row stays the only number (J = +0.36, one-sided p 0.337) |
| Fin Nifty, last hour | NOT COMPUTED | | | | | | | no Fin Nifty data (Section 7) |

In words: in the last hour on Bank Nifty the hybrid without the 17 option-market inputs beats the full hybrid by 2.4
points of mcsGARCH's QLIKE. The full hybrid's last-hour gain over mcsGARCH is 8.00 points and the `optionmkt` hybrid's
10.40, so the option-market inputs cost the hybrid 30% of its last-hour gain (`shares.csv`: share -0.30, 95% bootstrap
interval [-0.74, -0.05]). The interval does not reach -1, let alone +1, so "worse by at least 1 point without them"
is excluded (p 0.999) and the mirror claim "better by at least 1 point without them" falls just short (p 0.088 after
Holm over the one computed row). The estimate is not within seed noise (seed SD 0.38). The plan expected "not shown"
here for lack of power, not a 2.4-point effect in the opposite direction; this is a secondary row and changes no
primary verdict, but it is the largest option-market effect found in the study, and it is a cost, not a gain. A
likely reading, not tested here: the previous evening's option chain is stale by the last hour, and the LightGBM
correction that uses it overfits in that window.

**S4, pooled over the three indices.** NOT COMPUTED: the pooled test needs all three indices and Fin Nifty was not
run (`pooled.csv` is empty).

## 4. Robustness (the H1 TOST in other samples; `tests.csv`, rows `R1_*` to `R5_*`)

REAL DATA. Each family is Holm-adjusted over its computed rows. "Passes" means Holm p below 0.05.

| Family | Nifty 15 min: J, raw p | Nifty 30 min: J, raw p | Bank Nifty 15 min | Bank Nifty 30 min |
|---|---|---|---|---|
| R1 seed set 7 alone | +0.598, 0.252 | -0.062, 0.010 | -0.225, 0.004 | -0.228, 0.046 |
| R2 new seed sets 101 to 104 | +0.300, 0.044 | +0.085, 0.007 | -0.165, 0.004 | -0.198, 0.030 |
| R3 delta-method variance | +0.360, 0.069 | +0.056, 0.006 | -0.177, 0.003 | -0.204, 0.030 |
| R4 without 4 and 5 June 2024 | +0.273, 0.050 | -0.056, 0.005 | -0.200, 0.004 | -0.258, 0.040 |
| R5 without the 1% largest mcsGARCH-loss days | -0.106, 0.0001 | -0.208, 0.018 | -0.117, 0.001 | -0.157, 0.020 |

Holm over the four rows of each family (`verdict.csv`, columns `R1_holm` to `R5_passes`): on Nifty at 30 minutes and
Bank Nifty at 15 minutes all five robustness rows pass; on Nifty at 15 minutes only R5 passes (Holm 0.0005; R2 0.061,
R3 0.069, R4 0.079, R1 0.252); on Bank Nifty at 30 minutes only R5 passes (Holm 0.036; R3 0.060, R2 0.061, R4 0.079,
R1 0.092).

What this says about Nifty at 15 minutes: the "not shown" verdict is fragile in the other direction. Without the 7
days with the largest mcsGARCH loss (R5), the joint value turns slightly negative (-0.11 points) and the interval
shrinks enough to pass at 1 point with raw p 0.0001; without the election days (R4) or with the four new seed sets
only (R2) the raw p is at the 5% line. So the +0.36 and the wide interval are carried by a handful of extreme days.
The same holds for Bank Nifty at 30 minutes (R5 passes with raw p 0.020). The pre-registered verdicts stand ("not
shown either way"); the robustness rows are reported next to them, as the plan requires, and do not replace them.

## 5. Exploratory and descriptive

**E1, do the chain and VIX stand in for each other?** (REAL DATA, `tests.csv` rows `E1_*`, `contributions.csv`; no
adjustment.) I = J - C_implied - C_vix on the five-seed average. Nifty 15 min: J +0.36, C_implied +0.12, C_vix -0.01,
so I = +0.25 (two-sided p 0.012; seed-set I from -0.26 to +0.68). Nifty 30 min: J +0.06, C_implied -0.12, C_vix -0.12,
I = +0.30 (p 0.022; seed-set I from +0.15 to +0.50). Bank Nifty 15 min: J -0.18, C_implied -0.22, C_vix +0.05
(both from P110's committed daily losses on the same five-seed average), I = -0.01 (p 0.893); Bank Nifty 30 min:
J -0.20, C_implied -0.26, C_vix +0.06, I = -0.005 (p 0.977). A positive I means removing both groups costs more than
the sum of removing each, which is the pattern of two inputs that stand in for each other. On Nifty this is the
pattern at both horizons, as the plan expected for 15 minutes; on Bank Nifty there is no interaction at all: the
joint drop costs exactly what the two single drops add up to. It is exploratory and the Nifty size (about a quarter
of a point) is well inside the 1-point margin.

**E2, expiry days only** (`tests.csv` rows `E2_*`, TOST at 1 point on the index's expiry days): Nifty 15 min J +0.13,
p 0.075; Nifty 30 min J +0.88, p 0.436; Bank Nifty 15 min J -0.41, p 0.212; Bank Nifty 30 min J -0.45, p 0.279. No
expiry-day row passes (few days, wide intervals).

**J as a share of the hybrid's gain** (REAL DATA, `shares.csv`, 95% stationary-bootstrap interval, 10,000 draws, mean
block 10 days): Nifty 15 min 6.2% [-7.5%, +18.9%]; Nifty 30 min 0.7% [-9.3%, +10.4%]; Bank Nifty 15 min -4.3%
[-20.5%, +10.2%]; Bank Nifty 30 min -3.9% [-21.3%, +12.4%]. Every interval covers zero; none reaches 25% of the gain.

**J in every seed set** (REAL DATA, `contributions.csv`, `seed_repeats.csv`): Nifty 15 min +0.598, +0.155, +0.488,
+0.184, +0.374 (seed sets 7, 101, 102, 103, 104; the same numbers as P106's `seed_repeats.csv`); Nifty 30 min -0.062,
+0.045, +0.356, +0.060, -0.121; Bank Nifty 15 min -0.225, -0.086, -0.143, -0.206, -0.227; Bank Nifty 30 min -0.228,
-0.141, -0.265, +0.018, -0.403. The hybrid's gain over mcsGARCH on the five-seed average (`contributions.csv`,
`hybrid_gain_points`): Nifty 5.79 and 7.84 points, Bank Nifty 4.09 and 5.26; the `optionmkt` hybrid's own gain is
5.43 and 7.78 on Nifty and 4.27 and 5.47 on Bank Nifty. For comparison, P110's `mcsinfo` gap on the same Bank Nifty
sample is +0.10 and +0.19 points (`C_mcsinfo_points`).

## 6. Reproduction checks (`reproduction_check.csv`)

* mcsGARCH's daily QLIKE is identical in all five seed sets on Nifty and on Bank Nifty at 15 and 30 minutes
  (checked by the script; the script refuses otherwise).
* The new Bank Nifty `optionmkt` work files against P110's committed daily losses (10 checks, one per target and seed
  set): the same 660 test days in every file; mcsGARCH's and `mcsgarch_iv`'s daily QLIKE agree on every day to a
  largest relative difference of 5.6e-16 (404 to 489 of the 660 days bit-identical, the rest differing in the last
  binary digit), so the script's bit-for-bit test reports "not identical" and `analyze` was run with
  `--allow-mismatch` (Section 7, item 7). The benchmark, the days and the targets are those of P110's run.
* The checks that need P106's and P110's git-ignored work files (`data/processed/p106`, `data/processed/p110`) were
  **not run**: those folders are not on this machine, so the plan's fallback to the committed daily losses applies
  (plan Section 8). That covers: the Nifty daily losses recomputed from P106's work files, the rerun Nifty
  `optionmkt` job forecast by forecast against P106's work file, and the row-by-row comparison of the new Bank Nifty
  `optionmkt` work files with P110's `full` work files.
* **The rerun Nifty `optionmkt` job (15 minutes, seed set 7) reproduces P106's committed daily losses** (post hoc check
  by the agent, `reproduction_check_nifty_rerun_by_hand.csv`, since P106's work file is absent and the script's own
  forecast-by-forecast check reports "not run"): the same 792 test days; the `optionmkt` daily QLIKE agrees with the
  `optionmkt` column of `results/p106_gain_decomposition/daily_losses_nifty_next15_s7.csv` on every day to a largest
  relative difference of 4.4e-16 (579 of 792 days bit-identical), and `mcsgarch` and `mcsgarch_iv` likewise (573 and
  586 days bit-identical, largest relative difference 4.4e-16). A LightGBM refit that differed in even one tree would
  show differences many orders of magnitude larger, so P120's code path reproduces P106's `optionmkt` forecasts on
  Nifty. This check is not part of the registered code (it is labelled POST HOC in the file).
* **The new Bank Nifty last-hour `full` hybrid with seed set 7 reproduces the committed headline pipeline exactly**
  (`reproduction_check.csv`, row "BANKNIFTY last_hour seed 7 full hybrid vs pipeline_nse_banknifty/forecasts_last_hour.csv"):
  660 of 660 rows matched, and `y`, `mcsgarch`, `mcsgarch_iv` and `hybrid_base` are bit-identical on every row
  (largest relative difference 0). So the code path, the raw data downloaded today and this machine's library
  versions reproduce a committed LightGBM fit forecast by forecast. The last-bit differences in the 15- and 30-minute
  daily QLIKE above therefore arise in the within-day averaging (this machine's newer numpy and pandas), not in the
  forecasts; this was not traced further. mcsGARCH's daily QLIKE is also identical in all five last-hour seed sets.
* The Nifty last-hour check against P99 (seed sets 101 to 104) was not run because those jobs were not run.
* Raw inputs against `data/MANIFEST.csv` (`run_info.json`, `input_sha256`, and `data/processed/p120/inputs_BANKNIFTY.json`):
  1,889 files agree (`banknifty_1min.csv` and all 944 bhavcopy and 944 participant-OI days of the manifest), 2 differ
  and 2 are not in the manifest. The agent checked which: the 2 differing files are `index_daily.csv` and
  `india_vix.csv`, which the fresh download extended by one day (3 November 2026) past the manifest's version, and
  the 2 absent files are that same day's bhavcopy and participant-OI files. 3 November 2026 is after Bank Nifty's
  last day (22 April 2026), so every raw input inside Bank Nifty's range matches the manifest.

## 7. Deviations from the plan

All decided before the corresponding outcome was seen unless stated.

1. **Date and time budget.** The plan's deadline was 18:30 UTC on 3 November 2026; the study was not run that day.
   This run started at 01:57 UTC on 4 November 2026 with a hard budget of 70 minutes of wall clock (computing had to
   stop at 60 minutes; it finished at 47), on a 4-core machine shared with five other agents. The plan's Section 11 expected 45 to 75 minutes
   for everything with 2 workers, so the stopping rule (plan Section 10) was applied from the start: jobs cut from the
   end.
2. **Raw data.** The NSE raw files (`data/raw`, git-ignored) were being downloaded again while this run started
   (`logs_tmp/download_nse.log`); the Bank Nifty run could only begin when that download ended at 02:11:21 UTC. The agent checked that the
   download's 57 missed days are the same days in all three folders and are all NSE holidays, and the market loader
   found the committed run's 810 days to 22 April 2026 (the settings check passed).
   P106's and P110's git-ignored work files are not on this machine: the committed daily losses were used instead and
   the checks that need the work files are "not run" (the plan's own fallback).
3. **Fin Nifty not run (decided at 4 minutes in, before any outcome).** `scripts/fetch_finnifty_data.py` reads NSE's
   official Fin Nifty daily values from `data/raw/nse_full/index_close_full/` (P96's download), which is not on this
   machine and has no fallback in the script, and it would download about 770 days of F&O bhavcopy from NSE's archive
   (at the 1.2 seconds per day the Nifty and Bank Nifty download was taking) while that download was still using the
   archive. That could not finish inside the budget together with the 10 Fin Nifty jobs. Both Fin Nifty H1 rows, its
   S1, S2, S3 rows, the pooled S4 rows and the Fin Nifty parts of E1, E2 and the descriptive tables are NOT COMPUTED.
4. **Nifty last-hour jobs not run (the plan's cut order).** The plan cuts the Nifty last-hour jobs first, then Fin
   Nifty's, then Bank Nifty's. The Bank Nifty last-hour jobs did run (the CPU freed up; the five two-variant jobs took
   92 to 109 seconds each), so the S3 row for Bank Nifty is computed. The four Nifty last-hour jobs (seed sets 101 to
   104, two variants each) were not run, so S3 for Nifty is NOT COMPUTED, as the stopping rule says. While the Bank
   Nifty jobs were slow (the first four took 297 to 384 seconds each, against the plan's 55 to 85), the agent had
   prepared to stop the run after the ten 15- and 30-minute jobs; when the pace improved it let the run finish
   instead. No result had been seen when either decision was made (the first scratch analysis of Bank Nifty results
   was made after the tenth job, with the last-hour jobs already running).
5. **Holm over fewer rows.** With only four of the six H1 rows computed, the Holm adjustment in `tests.csv` and
   `verdict.csv` spans four rows (and each secondary and robustness family likewise spans its four computed rows),
   which is less strict than the plan's six. A reader can apply the plan's six-row Holm by hand: the step-down
   multipliers are 6, 5, 4 and 3 for the four ordered raw p-values when the two missing rows are assumed to have the
   largest p. Under that six-row Holm the two passing rows still pass (Bank Nifty 15 min: 0.0032 x 6 = 0.019; Nifty
   30 min: 0.0061 x 5 = 0.031) and the two failing rows still fail (Bank Nifty 30 min: 0.030 x 4 = 0.121; Nifty 15
   min: 0.075 x 3 = 0.226), so no H1 verdict depends on the number of rows. The raw p-values are given on every row so
   a reader can apply any correction.
6. **The Nifty stage-1 run was restricted to the reproduction job** (`--targets next15 --seeds 7`, one job, 1 worker)
   and started only after the Bank Nifty run had finished (02:38:02 UTC), because at most 2 worker processes were
   allowed in all. It took 5.3 minutes (market loaded in 47 s, datasets at 193 s, the fit 122 s) and finished at
   02:43:24 UTC, inside the compute window. The four Nifty last-hour jobs were not run (item 4). Because P106's work
   file is absent, the script's own forecast-level check reports "not run"; the agent added the post hoc daily-QLIKE
   check of Section 6 instead. The Nifty 15- and 30-minute rows do not depend on this job (they come from the
   committed daily losses). After this job, `analyze` was run a second time so that `timing.csv` and `run_info.json`
   include it; every result table was byte-identical to the first run (checked with `cmp`), and the first run's
   stdout is kept in the agent's scratch folder.
7. **`analyze` was run with `--allow-mismatch`** (decided after seeing the check's size, before any test result of Bank
   Nifty was seen). The script refuses by default unless mcsGARCH's daily QLIKE recomputed from the new Bank Nifty
   work files equals P110's committed daily losses bit for bit. Here the same 660 days are present and the largest
   relative difference on any day is 5.6e-16 (`reproduction_check.csv`, columns `mcsgarch_max_abs_rel_diff` and
   `mcsgarch_iv_max_abs_rel_diff`; 404 to 489 of 660 days are bit-identical, the rest differ in the last binary digit).
   This machine has numpy 2.5.4 and pandas 3.1.0, while P110 recorded numpy 2.5.3 and pandas 3.0.5
   (`results/p110_gain_stress_test/run_info.json`); the within-day averaging is where a last-bit difference of this
   size arises. The test rows use P110's committed `full` and `mcsgarch` columns and this study's `optionmkt` column,
   as the plan specifies, so the mismatch enters no number of this study. Whether the new 15- and 30-minute LightGBM
   fits are bit-identical to what P110's environment would have produced cannot be checked directly (no work files),
   but the two forecast-level checks that were possible both passed: the seed-7 last-hour fit reproduces the committed
   Bank Nifty pipeline bit for bit, and the Nifty rerun reproduces P106's committed daily losses to 4.4e-16
   (Section 6).
8. No number of a finished job was changed, no job was rerun to get a different answer, and no seed set was added or
   dropped. The daily losses of every finished Bank Nifty seed set are written to
   `results/p120_joint_option_market/daily_losses_banknifty_<target>_s<seed>.csv`.

## 8. Commands and run time (`run_info.json`, `timing.csv`)

```
OMP_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_p120_joint_option_market.py   # 9 passed
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py run --index BANKNIFTY --workers 2 \
    > results/p120_joint_option_market/run_stdout_banknifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py run --index NIFTY --workers 1 \
    --targets next15 --seeds 7 > results/p120_joint_option_market/run_stdout_nifty.txt 2>&1      # deviation, Section 7 item 6
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py analyze --plan-commit 30da69e \
    --allow-mismatch > results/p120_joint_option_market/analyze_stdout.txt 2>&1                   # deviation, Section 7 item 7
# post hoc daily-QLIKE check of the Nifty rerun (agent's script, not in the repository):
#   writes results/p120_joint_option_market/reproduction_check_nifty_rerun_by_hand.csv and _stdout.txt
```

Run time (REAL DATA, `run_info.json`, `timing.csv`, `run_stdout_banknifty.txt`, `run_stdout_nifty.txt`): the Bank
Nifty stage 1 started at 02:11:21 UTC on 4 November 2026 (market loaded in 46 s, panels and datasets built at 153 s)
and finished at 02:37:54 UTC, 26.5 minutes of wall clock with 2 worker processes on a 4-core machine shared with five
other agents; its 15 jobs took 2,802 seconds of job time (the ten 15- and 30-minute `optionmkt` jobs 108 to 384 s
each, slowest first while the machine was busiest; the five last-hour two-variant jobs 92 to 109 s each). The Nifty
reproduction job took 5.3 minutes of wall clock (122 s of fit time) from 02:38:02 to 02:43:24 UTC. All job time:
2,924 s. The analysis took 3.3 s. The synthetic tests took about 1.5 minutes. The Fin Nifty stage was not run. The
whole run, from the first command to the last analysis, took 47 minutes of the 70 allowed. `run_info.json` records
the plan commit, the analysis commit, the library versions (numpy 2.5.4, pandas 3.1.0, lightgbm 4.7.0, arch 8.0.0,
scipy 1.18.1, Python 3.13.18), the SHA-256 of every raw input and of every committed reference file; the full test
table has 52 rows (`tests.csv`).

## 9. Limitations

1. **Two of the six primary rows are missing** (Fin Nifty), and so are the Nifty and Fin Nifty last-hour rows and
   the pooled rows: this is a partial run of the registered plan, and the missing rows must be run before the plan's
   rules can be applied to all three indices. P110's single-group rows on Fin Nifty (plan, Prior knowledge) are the
   only guide there. The Bank Nifty last-hour finding (Section 3) has no counterpart on the other two indices yet.
2. **No new days.** The tests use the same test days as P106 and P110; Nifty's point estimates were public. The
   interval width comes mostly from which days happened (the plan's Section 7 and the R5 row above), so more seed
   sets would not narrow it much.
3. **The margin is a choice.** 1 point is about a sixth to a quarter of the hybrid's gain (5.8 and 7.8 points on
   Nifty) and about the size of Bank Nifty's time-of-day contribution in P110. The 0.5-point rows and the intervals let
   a reader use another margin; no computed row passes at 0.5 point.
4. **Removing a block of inputs measures what it adds given the others.** The 24 remaining inputs include mcsGARCH's
   own forecast and the realized history, which may already carry some of what the option market knows. "The hybrid
   does not need the option market" is what was tested; "the option market holds no information" was not.
5. **The reproduction checks that need the earlier work files were not run** (Section 6); the Bank Nifty run was
   checked against P110's committed daily losses instead (mcsGARCH's daily QLIKE day by day, agreeing to 5.6e-16),
   the seed-7 last-hour fit against the committed pipeline (bit-identical) and the Nifty rerun against P106's
   committed daily losses (agreeing to 4.4e-16); the Nifty 15- and 30-minute rows are computed from the committed
   files directly.

## 10. The sentence the project's documents may use

By the plan's Section 4 rules (fixed in advance), with four of six rows computed:

> No significant joint contribution of the option-implied and VIX inputs was found on any computed row, and a joint
> contribution of 1 point (1% of mcsGARCH's QLIKE) or more is ruled out only on Nifty at 30 minutes and Bank Nifty at
> 15 minutes; it is not ruled out on Nifty at 15 minutes (90% interval [-0.37, +1.09] points) or, after the Holm
> adjustment, on Bank Nifty at 30 minutes ([-0.90, +0.49]), and Fin Nifty was not tested.

P106's wording "removing VIX together with the chain does not reveal a hidden option-market contribution" must be
replaced by that sentence, and the unqualified "the gain does not come from option-market information" may not be
used for the three indices together. Wherever the documents discuss the last hour on Bank Nifty they should add
(secondary row S3, not a primary verdict): "in the last hour on Bank Nifty the hybrid does better without the
option-market inputs, by 2.4 points of mcsGARCH's QLIKE (two-sided p 0.020, all five seed sets agree), about 30% of
its last-hour gain".
