# P118 pre-registration: the hybrid built on mcsGARCH as the paper specifies it

**Status.** Written on 3 November 2026 (phase 1) by an AI agent (Claude) working for Shreyan Thanki. **This design
is committed to git before any forecast or loss of the new model (`hybrid_cpp`) has been computed on real data.**
No forecast of `hybrid_cpp` or of `hybrid_cpp_all` has been made on real data, and no loss, gain, test statistic or
bootstrap draw involving either has been computed on real data. The only real-data computations so far are:

1. the reproduction check of Section 14: this study's driver, with its new code path switched on, rebuilt the
   committed hybrid (seed set 7) and the committed benchmark on Nifty at all three targets, and also refitted
   `mcs_cpp` inside the same walk-forward; the rebuilt forecasts were compared number by number with the committed
   forecasts and with P116's `mcs_cpp` forecasts (forecasts and targets only, no loss computed);
2. planning numbers computed from already committed daily losses of committed models (P106, P110, P116; "Prior
   knowledge" below), to judge the power of the tests.

The script and its tests were also run end to end on simulated markets (SYNTHETIC). The commit that adds this file
is the time stamp; phase 2 (the run of `hybrid_cpp`) starts only after that commit exists and has been pushed. The
code that phase 2 runs (`experiments/p118_hybrid_on_paper_mcsgarch.py` and an optional code path in
`src/niftygex/forecast.py` whose defaults run the committed code) and its tests
(`tests/test_p118_hybrid_on_paper_mcsgarch.py`, synthetic data only) are in the same commit. Phase 2 writes
`results/p118_hybrid_on_paper_mcsgarch/` and `docs/P118_HYBRID_ON_PAPER_MCSGARCH.md`. All results will be REAL DATA.

## In plain words

The project's hybrid forecasts how much the Nifty will move in the next 15 or 30 minutes. It takes a forecast
from mcsGARCH (the anchor paper's model) and multiplies it by a correction that LightGBM (a machine-learning method)
learns from 41 inputs. The mcsGARCH forecast is used twice: as the starting point that the correction multiplies
(the "base"), and as one of the 41 inputs.

Study P116 found that the version of mcsGARCH the hybrid is built on (`mcs_ours`, our implementation) is not
exactly the paper's model, and that on Nifty the paper's own specification (`mcs_cpp`) forecasts slightly better,
by 1.45% and 1.09% of its loss at 15 and 30 minutes. P116 then compared the committed hybrid (still built on
`mcs_ours`) with `mcs_cpp`: the hybrid is 4.43% and 6.64% better.

That leaves an obvious question. **What if the identical recipe is built on the paper's model instead?** This study
builds that hybrid (`hybrid_cpp`): the same 41 inputs, the same LightGBM settings, the same walk-forward with a refit
every 21 trading days, the same seed sets, with `mcs_cpp`'s forecast in place of `mcs_ours`'s as the base and as
the input. It answers two questions:

1. **How much better is `hybrid_cpp` than the paper's model?** This gives the cleanest statement for the professor:
   "your model as you specify it, plus a learnt correction, is X% better than your model".
2. **Does the correction absorb the choice of base?** If it does, the two hybrids forecast equally well, and it does
   not matter which version of mcsGARCH the hybrid starts from. If it does not, one base gives a better hybrid.

## Why this study

* P116's write-up (`docs/P116_BENCHMARK_AS_PRINTED.md`, caveat 2) says: "A hybrid built on `mcs_cpp` might gain a
  different amount over `mcs_cpp`; that would be a new model and was not fitted." P116's plan (Section 10, weakness
  2) says the same. This study fits it.
* P116 found that part of what the hybrid's correction did at 14:45 and 15:00 was to undo our benchmark's 5-minute
  smoothing (its gain at 15:00 is 12.51% against `mcs_ours` but 6.68% against `mcs_cpp`). So the correction can
  absorb some of a base's flaws. Whether it absorbs all of them is the open question.
* The project's documents now quote the headline against `mcs_cpp` (4.43% and 6.64%). That is a gain of a hybrid
  built on one model over another model. A referee or co-author may ask for the like-for-like comparison: the
  paper's model against the paper's model plus a correction.

## Prior knowledge (what was known when this was written)

All REAL DATA, from committed files. Units: "points" are % of `mcs_cpp`'s mean daily QLIKE.

* **P116** (`results/p116_benchmark_as_printed/`, 792 Nifty test days, 10 August 2023 to 2 November 2026): mean
  daily QLIKE of `mcs_cpp` 0.2804 (15 minutes) and 0.2152 (30 minutes); `mcs_ours` is worse by 1.45 and 1.09
  points (two-sided DM, Holm p 0.005 and 0.014). On Bank Nifty `mcs_ours` is better (by 0.84 and 1.23 points), on
  Fin Nifty worse (0.91 and 0.97, not significant after Holm). The expiry-day curve and the smoothing carry the
  Nifty difference; the ARMA mean changes nothing that matters.
* **The committed hybrid against `mcs_cpp`, by seed set** (planning numbers computed from P106's committed Nifty
  daily losses, P110's committed Bank Nifty and Fin Nifty daily losses and P116's committed daily losses of
  `mcs_cpp`; gain in % of `mcs_cpp`'s QLIKE):

  | Index, target | B: `mcs_cpp`'s gain over `mcs_ours` | seed set 7 | 101 | 102 | 103 | 104 | daily QLIKE averaged over the 5 |
  |---|---|---|---|---|---|---|---|
  | Nifty, 15 min | 1.45 | 4.43 | 4.27 | 4.46 | 4.42 | 4.52 | 4.42 |
  | Nifty, 30 min | 1.09 | 6.64 | 6.81 | 7.01 | 6.94 | 6.76 | 6.83 |
  | Bank Nifty, 15 min | -0.84 | 4.73 | 4.90 | 4.90 | 5.01 | 4.97 | 4.90 |
  | Bank Nifty, 30 min | -1.23 | 6.20 | 6.41 | 6.41 | 6.70 | 6.39 | 6.42 |
  | Fin Nifty, 15 min | 0.91 | 4.03 | 3.84 | 4.00 | 4.03 | 3.95 | 3.97 |
  | Fin Nifty, 30 min | 0.97 | 5.18 | 5.08 | 5.12 | 5.14 | 5.15 | 5.13 |

  Against `mcs_ours` the five-seed average gains on Nifty are 5.79 and 7.84 (seed set 7: 5.80 and 7.65).
* **How precisely can two hybrids be compared?** (planning numbers, Nifty, committed daily losses; the standard
  error is the DM test's Newey-West standard error of the mean daily difference, in points; the TOST p is P110's
  `tost` at a 0.5-point margin):

  | Pair (committed models only) | 15 min: difference (SE), TOST p | 30 min: difference (SE), TOST p |
  |---|---|---|
  | the hybrid, seed set 101 vs seed set 7 (only the seeds differ) | +0.16 (0.10), below 0.001 | -0.17 (0.21), 0.056 |
  | seed set 102 vs 7 | -0.03 (0.12), below 0.001 | -0.36 (0.22), 0.266 |
  | seed set 103 vs 7 | +0.01 (0.13), below 0.001 | -0.30 (0.17), 0.116 |
  | seed set 104 vs 7 | -0.09 (0.14), 0.001 | -0.11 (0.12), 0.001 |
  | P106's `mcsinfo` hybrid vs the hybrid (other inputs, seed set 7) | -0.26 (0.31), 0.225 | -0.36 (0.42), 0.371 |
  | the hybrid without the calendar inputs vs the hybrid | -0.42 (0.26), 0.370 | -0.59 (0.31), 0.610 |
  | `mcs_ours` vs `mcs_cpp` (the two bases) | +1.45 (0.48), 0.976 | +1.09 (0.44), 0.907 |

  **What this means for the design.** Two hybrids that differ only in their LightGBM seeds differ by up to 0.36
  point at 30 minutes, and a 0.5-point TOST on one seed set each cannot always call them equivalent (3 of 4 pairs
  fail at 30 minutes). Hybrids that differ in their inputs have standard errors of 0.26 to 0.42. A TOST at 0.5
  point passes only if the estimate plus 1.65 standard errors stays below 0.5, so with a standard error of 0.25 the
  estimate would have to be within about 0.09 of zero. **The equivalence test has limited power, especially at 30
  minutes.** This is why the primary sample averages the daily losses over the five seed sets (Section 3), which
  removes most of the seed noise, and why the plan reports, for every TOST row, the smallest margin that would
  pass. The margin itself is not changed to gain power (Section 6 gives the reasons for 0.5).
* **P99 and P110**: the hybrid's gain moves by up to 0.28 points (15 minutes) and 0.35 points (30 minutes) across 20
  seed sets on Nifty; on Bank Nifty seed set 7 was the weakest of five.
* **No hybrid built on `mcs_cpp`, or on any mcsGARCH other than `mcs_ours`, has been fitted on real data.** The
  repository was searched for `alt_base`, `hybridalt` and hybrids with `expiry_diurnal=False` or a mean other than
  zero; there are none (every hybrid uses `WalkForwardConfig`'s committed `expiry_diurnal=True`,
  `diurnal_smooth=5` and a zero mean).

## 1. The models

| Name | Base (multiplied by the correction, and the input `log_mcs`) | Other inputs that come from mcsGARCH | Seed sets | Role |
|---|---|---|---|---|
| `mcs_ours` | | | | the committed benchmark (rebuilt and checked identical) |
| `mcs_cpp` | | | | mcsGARCH as the paper specifies it (P116's model, refitted inside every job and checked identical to P116's forecasts) |
| `hybrid` | `mcs_ours` | from `mcs_ours` and `mcs_iv` (as committed) | 7, 101, 102, 103, 104 | the committed hybrid (`hybrid_base`): seed set 7 from the committed forecasts, the other four from P106's (Nifty) and P110's (Bank Nifty, Fin Nifty) committed daily losses |
| **`hybrid_cpp`** | **`mcs_cpp`** | **from `mcs_ours` and `mcs_iv` (as committed)** | **7, 101, 102, 103, 104** | **the new model: the identical recipe on the paper's mcsGARCH** |
| `hybrid_cpp_all` | `mcs_cpp` | every one from the paper's specification | 7 | exploratory (E1) |
| `combination` | | | | P109's combination of five fixes, from its committed daily losses (S3) |

**`hybrid_cpp` in detail.** The committed hybrid predicts log(y) - log(mcs_ours) with LightGBM from the 41 inputs
of `hybrid_base` (P106's `HYBRID_COLUMNS`: the 37 inputs of `features.BASE_FEATURES` and `log_mcs`, `log_s_window`,
`log_mcs_iv`, `log_implied_window`), and forecasts mcs_ours x exp(prediction) x smearing factor. `hybrid_cpp` is
exactly this with `mcs_cpp`'s forecast of y in place of `mcs_ours`'s **in both places where the base forecast
enters**:

1. the multiplicative base (the offset of the log regression and the forecast's starting point);
2. the input `log_mcs` (in the same column position, so LightGBM sees the same 41 columns in the same order).

**Everything else is identical**: the 40 other inputs, including `log_s_window` (the share of `mcs_ours`'s diurnal
curve in the window), `log_mcs_iv` (the implied-level mcsGARCH, built with our settings) and `log_implied_window`;
the LightGBM settings (no tuning, 15 leaves, minimum leaf size 100, learning rate 0.03, rounds chosen by early
stopping on the last 20% of the training window, five LightGBM seeds per seed set averaged in logs); the refit before
every block of 21 trading days on all earlier days; the first 150 days used for training only; the targets, anchors
and test days. `mcs_cpp` is refitted at exactly the same refits, on the same daily level h (the same daily
AR(1)-EGARCH(1,1) fit), and its forecast of y is P116's: the variance part plus the expected squared ARMA(1,1) mean
(`McsGarch.forecast_variance` plus `arma11_mean_square_forecast`). As for the committed hybrid, the training rows'
base forecasts come from the current refit (fitted on the same training days), and the test rows' from the same
refit.

**Why only the base, not the other mcsGARCH-derived inputs.** The question is whether the choice of base forecast
matters, so the base is the one thing that changes. `log_s_window` and `log_mcs_iv` are inputs that carry time of
day and the option-implied level; changing them would change the information the correction sees, not only the base.
`hybrid_cpp_all` (exploratory E1) also changes them, to show whether that matters.

**`hybrid_cpp_all` in detail** (exploratory, seed set 7): every mcsGARCH of the walk-forward is fitted as the paper
specifies it (one diurnal curve, no smoothing, ARMA(1,1) mean with its expected squared mean in the forecast):
the base and `log_mcs` are `mcs_cpp` (identical to `hybrid_cpp`'s), `log_s_window` and the share behind
`log_implied_window` come from `mcs_cpp`'s one unsmoothed curve, and `log_mcs_iv` from the implied-level mcsGARCH
with the paper's settings. Everything else as `hybrid_cpp`.

## 2. Design

**Walk-forward.** The pipeline's own `run_walkforward` (`src/niftygex/forecast.py`) with the committed runs'
configuration (P106's `make_config`, which rebuilt the committed hybrid bit for bit in P106 and P110), plus the new
optional settings of Section 13. One job per index, target, seed set and kind; at most 2 worker processes.

**Jobs** (in this order; a finished job is not rerun):

| Index | Jobs | Purpose |
|---|---|---|
| Nifty | `hybrid_cpp` with seed set 7 at 15 min, 30 min and the last hour (3) | primary, S1 |
| Nifty | `hybrid_cpp` with seed sets 101 to 104 at 15 and 30 min (8; the first, seed set 101 at 15 min, also refits the committed hybrid as a check) | primary (five-seed average) |
| Nifty | `hybrid_cpp_all` with seed set 7 at 15 min, 30 min and the last hour (3) | exploratory E1 |
| Bank Nifty, Fin Nifty | `hybrid_cpp` with seed set 7 at 15 and 30 min, refitting the committed hybrid alongside as a check (2 each) | S4, S5 |
| Bank Nifty, Fin Nifty | `hybrid_cpp` with seed sets 101 to 104 at 15 and 30 min (8 each) | S4, S5 (five-seed average) |

**Days.** Nifty 942 days (2 January 2023 to 2 November 2026), 38 refits, **792 test days** (10 August 2023 to
2 November 2026; 19,800 forecasts at 15 minutes, 19,008 at 30 minutes, 792 in the last hour); 169 expiry days.
Bank Nifty 810 days to 22 April 2026, 660 test days. Fin Nifty 754 days to 27 January 2026, 604 test days. The
script refuses to run if the settings, the number of days or the last day differ from the committed run.

**Targets.** The sum of squared one-minute returns over the next 15 minutes (25 anchors a day, every 15 minutes from
09:15 to 15:15) and the next 30 minutes (24 anchors, 09:15 to 15:00), and over the last hour from 14:30 (one anchor;
trained on every 60-minute window), exactly as the committed runs. Bank Nifty and Fin Nifty: 15 and 30 minutes only.

**Seed sets.** The committed hybrid's LightGBM uses base seed 7 with five inner seeds (7, 1007, ..., 4007), averaged
in logs. Seed sets 101 to 104 (P99, P106, P110) replace the base seed. `hybrid_cpp` uses the same seed sets. The last
hour has only seed set 7 (the committed hybrid has no other seed set there).

## 3. Statistics

All losses are QLIKE, y/f - log(y/f) - 1, averaged within each day first (`niftygex.evaluation.daily_losses`);
every day has the same number of forecasts (checked).

* **Primary sample** (Nifty, 15 and 30 minutes, and S2, S4, S5): each hybrid's daily QLIKE **averaged over the five
  seed sets** (7, 101, 102, 103, 104), `hybrid` and `hybrid_cpp` alike. It is the recipe's expected loss with the
  LightGBM randomness averaged out. Seed set 7 alone (the literal headline model) is robustness R1. If the four
  repeat seed sets cannot be finished (Section 10), seed set 7 alone becomes the primary sample and this is reported.
* **G**: `hybrid_cpp`'s gain over `mcs_cpp`, 100 x (L_cpp - L_hybrid_cpp) / L_cpp, % of `mcs_cpp`'s QLIKE.
* **E**: `hybrid_cpp`'s gain over the committed hybrid, 100 x (L_hybrid - L_hybrid_cpp) / L_cpp, points of
  `mcs_cpp`'s QLIKE. **Positive means the hybrid built on the paper's model is better.** E equals G minus the
  committed hybrid's gain over `mcs_cpp`.
* **B**: the difference of the two bases, 100 x (L_ours - L_cpp) / L_cpp (1.45 and 1.09 on Nifty, Prior knowledge).
* **A = 1 - E / B**: the share of the difference between the bases that the correction absorbs. A = 1: the two
  hybrids are equally good (fully absorbed); A = 0: the hybrids differ by as much as their bases (nothing absorbed).
  Descriptive, with a 95% interval.
* **Own-base gain difference**: G minus the committed hybrid's gain over `mcs_ours` (each hybrid against the model
  it is built on). Descriptive.
* **Tests.** `diebold_mariano` from `niftygex.evaluation` (Newey-West variance, Harvey-Leybourne-Newbold factor,
  Student t with n - 1 degrees of freedom), as P106, P109, P110 and P116. **Equivalence**: P110's `tost` (two
  one-sided tests on the daily series 100 x (L_hybrid - L_hybrid_cpp) / mean L_cpp, whose mean is E, with the DM
  test's variance; p is the larger one-sided p; the 90% interval lies inside +-margin exactly when the test
  rejects at 5%). **Delta method** (R2): P110's `tost_delta_method` on the ratio, which also lets the denominator
  vary.
* **Intervals.** 95% stationary-bootstrap percentile intervals for E, B, A and the own-base gain difference (days
  resampled jointly for all four models, mean block 10 days, 10,000 draws, seed 118; `stationary_bootstrap_indices`
  and `bootstrap_means` of `experiments/referee_forecast.py`).
* **Descriptive tables**: every model's mean QLIKE and gains over `mcs_cpp` and `mcs_ours` on all days, expiry and
  non-expiry days and the three expiry regimes; E, B and A by index, target and subset; every seed set's G and E;
  the gains at every anchor minute (seed set 7); every TOST row's estimate, standard error, 90% interval and smallest
  passing margin.

## 4. Hypotheses and tests

**Primary family (4 tests, Holm over all four)**, Nifty, all 792 test days, five-seed average:

* **H1, `P_next15_h1` and `P_next30_h1`**: `hybrid_cpp` has lower mean QLIKE than `mcs_cpp`. One-sided DM.
  Estimate: G.
* **H2, `P_next15_tost` and `P_next30_tost`**: `hybrid_cpp` and the committed hybrid are equivalent: |E| < 0.5 point
  of `mcs_cpp`'s QLIKE. TOST. Estimate: E.

**Secondary families** (each with its own Holm adjustment; none can change the primary verdict, except that S0
enters the decision rule as stated in Section 6):

* **S0, the difference (2 tests)**: `S0_next15_diff`, `S0_next30_diff`: `hybrid_cpp` and the committed hybrid have
  different mean QLIKE (two-sided DM, Nifty, five-seed average). It tells the decision rule which way the base
  matters when equivalence is not shown.
* **S1, last hour (3 tests)**, seed set 7: H1, the TOST and the difference test for the 14:30 forecast.
* **S2, expiry days (6 tests)**: H1, TOST and difference at 15 and 30 minutes on the 169 expiry days (five-seed
  average). P116 found the expiry-day curve hurts `mcs_ours` most there.
* **S3, P109's combination (2 tests)**: `hybrid_cpp` has lower mean QLIKE than P109's combination at 15 and 30
  minutes (one-sided; P109 found the committed hybrid does not beat the combination).
* **S4, Bank Nifty (6 tests)** and **S5, Fin Nifty (6 tests)**: H1, TOST and difference at 15 and 30 minutes
  (five-seed average), nothing retuned. The decision rule is applied to them descriptively.

**Robustness** (each with its own Holm adjustment):

* **R1, seed set 7 alone (6 tests)**, Nifty: H1, TOST and difference at 15 and 30 minutes with the headline seed set
  only (the literal "identical recipe": seed set 7 against seed set 7).
* **R2, delta-method TOST (2 tests)**, Nifty, five-seed average.

**Placebo** (each with its own Holm adjustment):

* **PL1 (8 tests)**: `hybrid_cpp` with seed set s against `hybrid_cpp` with seed set 7 (s = 101 to 104), TOST at 0.5,
  Nifty, 15 and 30 minutes. Only the seeds differ, so these show how often the TOST can call two versions of one
  model equivalent on one seed set each.
* **PL0 (8 tests)**: the same for the committed hybrid. **These were computed during planning** on committed losses
  (Prior knowledge: at 30 minutes three of four fail even before adjustment); they are listed so that every p-value
  of the study is in `tests.csv`, and they are labelled as known in advance.

**Exploratory** (no adjustment; raw p, labelled exploratory):

* **E1, `hybrid_cpp_all` (9 rows)**, seed set 7, Nifty, three targets: it beats `mcs_cpp` (one-sided); it differs
  from `hybrid_cpp` (two-sided); it differs from the committed hybrid (two-sided).
* **E2, non-expiry days (6 rows)**: H1, TOST and difference on the 623 non-expiry days.
* **E3, the open and the rest of the day (4 rows)**, seed set 7: the difference test at the 09:15 anchor only and at
  the other anchors (P116 found the two bases differ most at the open and at 14:45 and 15:00).
* **E4, against our version (3 rows)**: `hybrid_cpp` beats `mcs_ours` (one-sided), 15 and 30 minutes (five-seed
  average) and the last hour (seed set 7).

Every p-value goes in `results/p118_hybrid_on_paper_mcsgarch/tests.csv` with the columns of P106, P110 and P116
(test_id, hypothesis, role, sample, estimate, statistic, p_raw, sidedness, p_method, p_adjusted, adjust_method), so
that the project's test ledger can read it. Any test added after this commit is labelled "posthoc".

## 5. The seed rule

As in P110, a row with five seed sets **passes only if** its Holm-adjusted p is below 0.05 **and** every seed set,
taken alone, points the same way:

* H1: each seed set's `hybrid_cpp` has a lower mean QLIKE than `mcs_cpp`;
* TOST: each seed set's E (seed set s of `hybrid_cpp` against seed set s of the committed hybrid) lies inside
  -0.5 to +0.5;
* difference: each seed set's E has the sign of the five-seed E.

A row whose Holm p passes but whose seed sets disagree is labelled **"seed-fragile"** and does not count as passing.
Rows on seed set 7 alone (R1, S1, E1, E3) have no seed rule.

## 6. Decision rule (fixed now; applied by `p118_verdict` in the script)

**At each primary horizon** (15 and 30 minutes), with the H2 (TOST) row and the S0 (difference) row, each labelled
by its Holm p and the seed rule:

* **"correction absorbs the base"**: the TOST row passes (equivalence within 0.5 point shown). If the S0 row also
  passes, a note says that a small difference, within the margin, favours one base.
* **"base matters, cpp better"**: the TOST row does not pass, the S0 row passes, and E > 0 (the hybrid built on the
  paper's model is better).
* **"base matters, ours better"**: the TOST row does not pass, the S0 row passes, and E < 0.
* In both "base matters" cases, if |E| is at most 0.5 the outcome carries the note "the point estimate is within the
  0.5-point margin, but equivalence is not shown".
* **"inconclusive"**: neither row passes (with a note if a row was only "seed-fragile").

**Overall**: if both horizons give the same outcome (ignoring notes), that is the verdict; otherwise "mixed", naming
each horizon's outcome. **Separately**, H1 is read as "`hybrid_cpp` beats `mcs_cpp`" at a horizon if its row passes.

**Why 0.5 point.** (1) It is P116's margin for "no material difference" of the headline, so the two studies speak
the same language. (2) It is a third to a half of the difference between the two bases on Nifty (1.45 and 1.09
points): equivalence within 0.5 point means the correction absorbs at least about half of that difference, while a
margin of 1 point would be about as large as the whole difference at 30 minutes and could not tell "absorbed" from
"passed through". (3) It is under a tenth of the hybrid's gain (4.4 to 7.8 points), so the wording of any headline
would not change within it. (4) It is fixed before any result. Its cost is power (Prior knowledge); the plan accepts
that, averages over seed sets to recover some, and reports the smallest passing margin of every TOST row so that a
reader can apply another margin.

**What we expect** (stated in advance; it changes no rule):

* **H1 holds at both horizons.** `hybrid_cpp`'s gain over `mcs_cpp` should lie between the committed hybrid's gain
  over `mcs_cpp` (4.42 and 6.83, five-seed average) and that value plus B (about 5.9 and 7.9): the first if the
  correction absorbs the base completely, the second if the base's advantage passes straight through.
* **E is small and positive (the paper's base a little better), most likely between 0 and +0.8 point at 15 minutes
  and between 0 and +0.5 at 30 minutes.** Reason: the smoothing flaw of `mcs_ours` (P116: the 14:45 and 15:00
  windows) is the same on every day, in the training rows and in the test rows, so the correction can learn it from
  the time-of-day inputs; that part should be absorbed (and P116 saw the committed hybrid undo part of it). The
  expiry-day curve's flaw is different: it is noise from estimating a curve on few days, and in the training rows the
  curve is fitted to those same days, so the training residuals understate the noise the test days will see; the
  correction cannot learn to undo it. So we expect A between about 0.4 and 1, and the expiry days (S2) to show more
  of the base difference than the other days.
* **The outcome.** Given the limited power, we expect "inconclusive" at 30 minutes; at 15 minutes "correction
  absorbs the base" if E is close to zero and the five-seed average is precise, otherwise "inconclusive" or "base
  matters, cpp better". On Bank Nifty, where `mcs_ours` is the better base, we expect E slightly negative.
* **`hybrid_cpp_all`** should be close to `hybrid_cpp` (the time-of-day inputs carry the same information whichever
  curve they come from).

This is a guess, not a rule.

## 7. What the documents will say under each outcome

The lead updates the project's summary documents; this study's write-up states the sentence to use.

* **H1 holds at both horizons** (expected): "Built on mcsGARCH exactly as the paper specifies it (one diurnal curve,
  no smoothing, ARMA(1,1) mean, Student t), the same hybrid recipe is G% and G% better than the paper's model at 15
  and 30 minutes (P118, five seed sets)." This is the like-for-like statement for the professor. If H1 fails at a
  horizon, the documents must say that the correction does not beat the paper's model at that horizon when built on
  it.
* **"correction absorbs the base"**: "The correction absorbs the choice of base: hybrids built on our implementation
  and on the paper's specification are equivalent within 0.5 point (P118). The hybrid's gain can be described as
  the paper's model plus a learnt correction." The committed hybrid stays the project's model.
* **"base matters, cpp better"**: "Built on the paper's specification the hybrid is E points better than the
  committed one (P118); part of the base's flaws passes through the correction." The documents quote `hybrid_cpp`'s
  gain over `mcs_cpp` as the like-for-like figure. Whether to make `hybrid_cpp` the project's model is a decision
  for Shreyan, taken after this result, and must be described as such (it would be a choice made after seeing the
  data, on the same days).
* **"base matters, ours better"**: "The committed hybrid is better than the same recipe built on the paper's model
  (P118)." The documents quote both gains and do not describe the hybrid as "the paper's model plus a correction"
  without that qualification.
* **"inconclusive"** or **"mixed"**: both hybrids' gains over `mcs_cpp` are quoted side by side with E and its
  interval, and the documents say the data cannot tell whether the base matters (or at which horizon).
* **A** and the expiry-day split (S2, E2) are reported in every case, to say where the base matters.

## 8. The exact commands

From the repository root, at the commit that adds this file, with the raw data listed in `data/MANIFEST.csv` in
`data/raw`:

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py run --index NIFTY --workers 2 \
    > results/p118_hybrid_on_paper_mcsgarch/run_stdout_nifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py run --index BANKNIFTY --workers 2 \
    > results/p118_hybrid_on_paper_mcsgarch/run_stdout_banknifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py run --index FINNIFTY --workers 2 \
    > results/p118_hybrid_on_paper_mcsgarch/run_stdout_finnifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py analyze --plan-commit <this commit> \
    > results/p118_hybrid_on_paper_mcsgarch/analyze_stdout.txt 2>&1
```

`run` writes one forecast file and one parameter file per job to `data/processed/p118/` (git-ignored) and the
SHA-256 of every raw input. `analyze` reads P116's `mcs_cpp` forecasts from `data/processed/p116_ref/` (a copy of
P116's git-ignored block files, taken before this plan was committed; see Section 14) and writes `tests.csv`,
`tests_detail.csv`, `verdict.csv`, `losses.csv`, `absorption.csv`, `seed_repeats.csv`, `equivalence_bounds.csv`,
`gain_by_anchor.csv`, `reproduction_check.csv`, `params_by_job.csv`, `daily_losses_<index>_<target>.csv` (every
model's daily QLIKE, from which every number can be recomputed), `timing.csv` and `run_info.json` (plan commit,
settings, input hashes, run time, versions).

## 9. Checks inside the run

`analyze` refuses to run unless every one of these holds (unless told to report a failure, which would be a
deviation):

* In every job, `y`, the rows (day, anchor, date), `mcsgarch` (`mcs_ours`) and `mcsgarch_iv` equal the committed
  forecasts on every row (for `hybrid_cpp_all`, `mcsgarch` must instead equal `mcs_cpp`).
* `mcs_cpp` (`mcsgarch_alt`) equals P116's `mcs_cpp` forecasts on every row, and is the same in every seed set's job.
* The committed hybrid refitted alongside (`cpp_base` jobs and the phase-1 `repro` jobs) equals the committed
  forecasts (seed set 7) or the committed daily losses (seed set 101 on Nifty) exactly.
* The committed daily losses used for seed sets 101 to 104 (P106, P110) and for P109's combination sit on the same
  days: their daily QLIKE of `mcs_ours` equals this study's on every day; P116's committed daily QLIKE of `mcs_ours`,
  `mcs_cpp` and the hybrid equal this study's.
* Every day has the same number of forecasts.

## 10. What will be reported, whatever the result, and the deviations policy

In `docs/P118_HYBRID_ON_PAPER_MCSGARCH.md`, for a beginner reader, every number labelled REAL DATA and taken from a
file in `results/p118_hybrid_on_paper_mcsgarch/`: the plain-language answer first; the verdict under the rule; the
main table (QLIKE of every model at 15 and 30 minutes and in the last hour, G, E, B and A with intervals); the seed
sets one by one; expiry days, the last hour, the combination, Bank Nifty and Fin Nifty; the robustness, placebo and
exploratory rows with their labels; what the result means for how the project describes the hybrid; the
reproduction checks; the run time; a "Deviations from the plan" section; caveats; files; and how to reproduce.

* After this file is committed the design does not change. Any change is a deviation, listed with its reason and
  with whether any outcome had been seen when it was decided. No other variant of the hybrid is fitted in this study.
* A crash is fixed with the smallest change that lets the run finish, and only if it changes no number of a job that
  already finished.
* **Order and deadline.** Nifty runs first (primary), then Bank Nifty, then Fin Nifty. If the repeat seed sets (101 to
  104) of Bank Nifty or Fin Nifty are not finished by 18:00 UTC on 3 November 2026, S4 or S5 is analysed on seed set
  7 alone and this is reported; if seed set 7 of an index is not finished by then, that index is dropped and this is
  reported. If Nifty's repeat seed sets are not finished by then, the primary sample becomes seed set 7 alone.

## 11. Expected run time

From the phase-1 reproduction run on real data (Section 14, `data/processed/p118/timing.csv`): loading the Nifty
market took 62 seconds and building the panels and datasets 189 seconds, so every index's run starts with about 4
minutes of loading. A Nifty job with one hybrid and the three mcsGARCH fits took 196 seconds (15 minutes), 178
seconds (30 minutes) and 144 seconds (last hour), two jobs in parallel. A job with two hybrids (`cpp_base`) should
take about 100 seconds more, and a `hybrid_cpp_all` job about as long as a `hybrid_cpp` job.

* **Nifty**: 14 jobs, about 2,650 seconds of job time, so about 25 to 30 minutes on 2 workers.
* **Bank Nifty and Fin Nifty**: 10 jobs each on fewer days, about 15 to 20 minutes each.
* **In all about 1 hour**, up to about 1.5 hours if the machine is busier; the analysis about 1 to 2 minutes.

The deadline of Section 10 (18:00 UTC) leaves a wide margin.

## 12. Known weaknesses

1. **Same days.** The 792 Nifty test days are those on which the headline and P116 were found; this study changes
   the base of the hybrid, not the data. Only later days can confirm it.
2. **Limited power for equivalence** (Prior knowledge). "Inconclusive" is a likely outcome and would mean the data
   cannot tell, not that the base matters.
3. **"As the paper specifies"** carries P116's caveats: the session grid, the two-step ARMA estimation, the constant
   in the daily AR(1) and the cross-day recursion are our choices.
4. **The other mcsGARCH-derived inputs stay ours in `hybrid_cpp`** (`log_s_window`, `log_mcs_iv`,
   `log_implied_window`). This is deliberate (Section 1); `hybrid_cpp_all` (exploratory, one seed set) shows whether
   it matters.
5. **Seed averaging** removes most of LightGBM's randomness from the primary sample but the DM, TOST and bootstrap do
   not include the remaining seed noise; the seed rule guards against a result carried by one seed set.
6. **No tuning.** Both hybrids use the committed, untuned settings. A hybrid tuned for `mcs_cpp` might do better or
   worse.

## 13. Code

* `src/niftygex/forecast.py`, an optional code path; with the defaults every model runs exactly the committed code:
  * `WalkForwardConfig.alt_base` (default `None`): keyword arguments of a second `McsGarch`, refitted at the same
    refits on the same daily level; its forecast of y (variance part plus expected squared conditional mean) is the
    column `mcsgarch_alt`, and the new model kind `hybridalt_<set>` is `hybrid_<set>` with log(`mcsgarch_alt`) in
    place of `log_mcs`, both as the offset and as the input in the same position;
  * `WalkForwardConfig.mcs_mean` (default `"zero"`): the intraday mean of every `McsGarch` of the run; a non-zero mean
    adds the expected squared conditional mean to each forecast (used only by `hybrid_cpp_all`);
  * `LEARNED_KINDS` gains `"hybridalt"`, which matches only names that start with `hybridalt_`.
* `experiments/p118_hybrid_on_paper_mcsgarch.py`, new: `run` (the jobs of Section 2, `--phase repro` for the check of
  Section 14), `check` (forecast-level reproduction checks, no loss), `analyze` (Sections 3 to 6). It reuses P106's
  configuration, work-file helpers, settings check and input hashing; P109's daily-loss frames; P110's TOST,
  delta-method TOST and DM pieces; P116's specification of `mcs_cpp` and, for the synthetic cross-check, its driver.
  `--synthetic` runs every stage on two simulated markets (labelled SYNTHETIC).
* `tests/test_p118_hybrid_on_paper_mcsgarch.py`, new, synthetic data only (13 tests): the defaults and model names
  are unchanged; asking for the new models without `alt_base` is an error; with the hook on, `mcsgarch`, `mcsgarch_iv` and `hybrid_base` are bit for bit the same as without
  it; with the committed specification as the second model, `hybridalt_base` equals `hybrid_base` exactly; the
  hybridalt inputs are the 41 committed columns with `log_mcs` replaced in place; `mcsgarch_alt` equals direct fits
  of the paper's specification and P116's own driver bit for bit; the `hybrid_cpp_all` settings give `mcs_cpp` as
  `mcsgarch`; no look-ahead (changing every return after a block leaves that block's forecasts unchanged); the job
  plan; E, B, A and the units and sign of the TOST row; the seed rule; the decision rule; and an end-to-end dry run of
  run, check and analyze on two simulated markets.

## 14. Checks run before this file was committed (3 November 2026)

* **The committed hybrid and P116's `mcs_cpp` reproduce exactly through this study's driver, with the new code path
  switched on** (REAL DATA, forecasts and targets only, no loss computed). Run from 12:00:32 to 12:05:54 UTC:

  ```
  PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py run --index NIFTY --phase repro --workers 2
  PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py check --indexes NIFTY
  ```

  Each `repro` job fitted `mcs_ours`, the implied-level mcsGARCH and `mcs_cpp` at every refit and the committed
  hybrid (seed set 7) on top of `mcs_ours`, but no `hybrid_cpp`. Every row is identical to the last bit (largest
  relative difference 0): `y`, the rows, `mcsgarch` (`mcs_ours`), `mcsgarch_iv` and `hybrid_base` against the
  committed forecasts in `results/pipeline_nse_v2/`, and `mcsgarch_alt` against P116's `mcs_cpp` forecasts: 19,800 of
  19,800 rows at 15 minutes, 19,008 of 19,008 at 30 minutes and 792 of 792 in the last hour
  (`data/processed/p118/forecast_checks.csv`). So the new code path leaves the committed hybrid untouched and refits
  exactly P116's `mcs_cpp` inside the walk-forward. These three files stay in the work folder; phase 2 does not
  rerun them and its `reproduction_check.csv` repeats these comparisons.
* **P116's `mcs_cpp` forecasts.** P116's git-ignored block files were in the P116 agent's own working copy. Before
  this plan was committed (11:44 UTC) the 99 block files (38 Nifty, 32 Bank Nifty, 29 Fin Nifty) were copied into
  `data/processed/p116_ref/` (git-ignored), the default `--p116-ref` folder. Only their forecasts are read.
* **Planning numbers** (Prior knowledge) were computed with two short scratch scripts (not committed) from the
  committed daily losses of committed models only: `results/p106_gain_decomposition/daily_losses_nifty_<target>_s<seed>.csv`,
  `results/p110_gain_stress_test/daily_losses_<index>_<target>_s<seed>.csv` and
  `results/p116_benchmark_as_printed/daily_losses_<index>_<target>.csv`. They checked that P106's and P110's daily
  QLIKE of mcsGARCH equals P116's `mcs_ours` on every day.
* **Dry run** (SYNTHETIC: two simulated markets of 120 days, refits every 20 days after 60, one LightGBM seed per
  seed set, seed sets 7 and 101): all stages ran; with the code path on, the committed hybrid equalled the pipeline's
  own walk-forward on every row, and `mcsgarch_alt` equalled P116's own driver on every row (73 reproduction rows,
  all identical). Each job took 1 to 2 seconds.
* **Tests.** `tests/test_p118_hybrid_on_paper_mcsgarch.py` (SYNTHETIC only): 13 passed in about 40 seconds. The
  existing tests of every module this study touches or loads (`tests/test_forecast.py`, `test_forecast_v2.py`,
  `test_p106_gain_decomposition.py`, `test_p109_mcsgarch_extensions.py`, `test_p110_gain_stress_test.py`,
  `test_p116_benchmark_as_printed.py`, `test_seed_bootstrap.py`, `test_volatility.py`, `test_demo_forecast_day.py`)
  together with this study's: 104 passed in 158 seconds. Every new or changed Python file compiles under Python 3.11
  (`python3.11 -m py_compile`), as the continuous-integration runs require. The full suite is run before the phase-2
  commit.
