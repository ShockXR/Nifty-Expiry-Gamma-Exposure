# P110: do P106's claims about the hybrid's gain hold on Bank Nifty and Fin Nifty?

Study P110, run by an AI agent (Claude) working for Shreyan Thanki on 3 November 2026.
**All numbers below are REAL DATA** (Bank Nifty and Fin Nifty, NSE data as in the headline runs; Nifty only through
P106's committed daily losses) unless marked SYNTHETIC. Every number comes from a file in
`results/p110_gain_stress_test/` (file named next to each table) or, where marked, from an earlier committed file.

* Registered plan: `docs/prereg/P110_GAIN_STRESS_TEST.md`, commit `5128e5365078000256ae2357f1ce7f7981fc1895`
  (committed 08:33:51 UTC and pushed before any forecast or loss of this study was computed; the forecasts started
  at 08:35:01 UTC).
* Code: `experiments/p110_gain_stress_test.py`, which reuses P106's code for the walk-forward, the work files and
  the contributions. Tests (SYNTHETIC only): `tests/test_p110_gain_stress_test.py`.
* Every p-value of the study: `results/p110_gain_stress_test/tests.csv` (316 rows).

*Note added 3 November 2026 after study P116:* here "mcsGARCH" and "the anchor paper's model" mean our
implementation (a separate time-of-day curve for expiry days, a 5-minute smoothing of the curves, a zero mean), not
the paper's own specification. Against the paper's specification (one curve, no smoothing, ARMA(1,1) mean) see
`docs/P116_BENCHMARK_AS_PRINTED.md`; there the hybrid's gain at 15 and 30 minutes is 4.43% and 6.64% on Nifty, 4.73%
and 6.20% on Bank Nifty and 4.03% and 5.18% on Fin Nifty. The contributions and shares below are measured in points
of our implementation's QLIKE and against its gain, and were not recomputed in P116.

## 1. The answer in plain words

Our hybrid forecasts how much an index will move over the next 15 or 30 minutes. It starts from the anchor paper's
model (mcsGARCH) and multiplies its forecast by a correction learnt by LightGBM from 41 inputs. Study P106 refitted
the hybrid without each group of inputs in turn and drew three conclusions:

1. the gain comes from recent realized volatility and the time of day;
2. the option-implied inputs, India VIX and the calendar inputs add nothing;
3. a hybrid that sees only mcsGARCH's own kind of information (the `mcsinfo` hybrid) gets the whole gain.

On Bank Nifty and Fin Nifty P106 had used only one set of random seeds, had never run the `mcsinfo` hybrid, and had
supported claim 2 only with tests that *failed to find* an effect. Failing to find an effect is not evidence that
there is none. So this study refitted everything on Bank Nifty (660 test days, to 22 April 2026) and Fin Nifty (604
test days, to 27 January 2026) with four new seed sets (101 to 104) as well as the headline seed set 7, added the
`mcsinfo` hybrid, and tested claim 2 with an **equivalence test**: a group "adds nothing" only if the data rule out
that it is worth 1 point or more in either direction. One point is 1% of mcsGARCH's average loss (QLIKE). The main
tests use the losses averaged over the four new seed sets. Nothing was retuned.

**The short answer** (REAL DATA, `claims.csv`, by the rules fixed in the plan before the run):

| P106's claim | Bank Nifty | Fin Nifty |
|---|---|---|
| 1. Realized volatility and the time of day carry the gain | **survives** | **survives** |
| 2. The option-implied, VIX and calendar groups add nothing | **qualified**: the calendar group adds a little at 30 minutes, under the margin | **qualified**: the calendar group adds a little at 15 minutes, under the margin |
| 3. The `mcsinfo` hybrid gets the whole gain | **partly survives** | **partly survives** |
| All three together | not all three survive | not all three survive |

**Claim 1 holds firmly on both indices** (REAL DATA, `verdict.csv`, `shares.csv`).

* Removing the **realized-volatility history** raises the loss by 1.79 and 2.16 points on Bank Nifty (15 and 30
  minutes) and by 1.73 and 2.05 points on Fin Nifty. That is 43% and 41% of the hybrid's gain on Bank Nifty and
  36% and 34% on Fin Nifty.
* Removing the **time of day** raises it by 1.28 and 1.16 points on Bank Nifty and 1.47 and 1.50 points on Fin
  Nifty (31% and 22%, and 31% and 25% of the gain).
* All eight rows have a Holm-adjusted p below 0.001, and all five seed sets agree.

**Claim 2 holds for the option market and VIX, with one weak spot, but not quite for the calendar** (REAL DATA,
`verdict.csv`, `equivalence_bounds.csv`).

* **Option-implied inputs and VIX.** Equivalence within 1 point is shown at both horizons on both indices. Removing
  either group changes the loss by between -0.22 and +0.14 points. The widest 90% interval (the option-implied group
  on Fin Nifty at 30 minutes) still ends 0.83 points from zero.
* **The weak spot: the option-implied group at 30 minutes.** It passes, but only just (Holm p 0.035 on both
  indices). It fails under the stricter single Holm adjustment over all 28 primary rows (R6), and in most other
  robustness samples (Section 7). With a 0.5-point margin the option-implied group fails at every index and
  horizon. So what the data show is "the option-implied group is worth less than about 0.8 point"; "worth nothing"
  is too strong.
* **Calendar inputs.** They are equivalent to zero within 1 point everywhere. But on Bank Nifty at 30 minutes
  (+0.27 points, Holm p 0.046) and on Fin Nifty at 15 minutes (+0.14 points, Holm p 0.046) removing them
  significantly *raises* the loss, with all five seed sets agreeing. The effect is small (the 90% intervals end at
  0.44 and 0.23 points), but it goes the other way from Nifty, where removing the calendar group made the hybrid
  slightly better. Under the plan's rule this makes claim 2 "qualified" on both indices. The Bank Nifty row is
  flagged "within seed noise" (its estimate is smaller than twice the spread across seed sets).

**Claim 3: the `mcsinfo` hybrid gets most of the gain, but "as good as the full hybrid, within 1 point" is shown
only on Bank Nifty at 15 minutes** (REAL DATA, `verdict.csv`, `contributions.csv`, `shares.csv`).

* It beats mcsGARCH everywhere: its loss is 4.00% and 5.09% lower than mcsGARCH's on Bank Nifty and 4.17% and 5.53%
  lower on Fin Nifty. The full hybrid's is 4.13% and 5.32%, and 4.83% and 6.03% lower. Holm p below 0.001, all five
  seed sets agree.
* Compared with the full hybrid it is worse by 0.13 and 0.23 points on Bank Nifty and by 0.65 and 0.50 points on Fin
  Nifty. Non-inferiority (at most 1 point worse) is shown only on Bank Nifty at 15 minutes (upper bound 0.57
  points). It is not shown on Bank Nifty at 30 minutes (upper bound 0.83, Holm p 0.055) or on Fin Nifty (upper
  bounds 1.17 and 1.33).
* On Fin Nifty every seed set puts `mcsinfo` behind the full hybrid (by 0.56 to 0.76 points at 15 minutes and 0.42
  to 0.68 at 30), yet the full hybrid's edge is not significant (Holm p 0.076 and 0.481). So the information mcsGARCH
  lacks (option market, VIX and most calendar inputs together) may be worth about half a point on Fin Nifty (14% and
  8% of the gain), and the data cannot rule out more than 1 point.
* In plain words: on every index the gain does not need the option market, because `mcsinfo` gets most of it. But
  "gets the whole gain" is Nifty's result and should not be stated for Fin Nifty.

**Pooled and Nifty** (REAL DATA, `pooled.csv`, `equivalence_bounds.csv`). Pooled over the three indices, or over
Bank Nifty and Fin Nifty (604 common days each, with errors that let the indices move together), every registered
pooled test passes. That includes
equivalence for the three groups and non-inferiority of `mcsinfo` (0.24 and 0.32 points behind the full hybrid in
the three-index pool). Re-analysing P106's
committed Nifty losses with the new tests, all equivalence and non-inferiority rows pass too. These pooled results
lean on Nifty, where `mcsinfo` was better than the full hybrid.

**A by-product: seed set 7 was an unlucky draw on Bank Nifty** (REAL DATA, `contributions.csv`, placebo PL1).

* On Bank Nifty the headline hybrid (seed set 7) had the smallest gain of the five seed sets at both horizons:
  3.92 and 5.03 points, against 4.09 to 4.21 and 5.23 to 5.55 for the new seed sets.
* On Fin Nifty seed set 7 had the largest gain at both horizons, by a small amount.
* The seeds alone moved the hybrid's Bank Nifty loss by 0.17 to 0.51 points.
* The registered headline numbers do not change.

## 2. What was registered (in short)

The plan (`docs/prereg/P110_GAIN_STRESS_TEST.md`) was committed and pushed before any forecast of this study was made
(commit 5128e53, 3 November 2026, 08:33 UTC; the forecasts started at 08:35 UTC). In short:

* **What was refitted.** On Bank Nifty and Fin Nifty, at 15 and 30 minutes, these variants were refitted:
  * the full hybrid;
  * the hybrid without each of P106's five input groups;
  * the `mcsinfo` hybrid. It keeps only past one-minute returns, the time of day and the expiry-day flag, which is
    what mcsGARCH itself uses.

  Seed sets 7 (the headline), 101, 102, 103 and 104 were used. The walk-forward, settings and test days were the
  same as in the committed runs; nothing was retuned. Nifty was not refitted: where it enters, P106's committed daily
  losses are used.
* **The primary sample** is each model's daily QLIKE averaged over the four **new** seed sets (101 to 104). Seed set
  7 was left out of it because its results on these indices were already public (P106).
* **Units.** "Points" are points of mcsGARCH's mean daily QLIKE: 1 point is 1% of mcsGARCH's average loss.
  * A group's contribution is 100 x (loss without the group - loss of the full hybrid) / mcsGARCH's loss.
  * Positive means the group helps.
  * The hybrid's gain in points is the same number as its margin over mcsGARCH in %.
* **Three primary families, one per P106 claim, each with its own Holm adjustment:**
  * **F1 (8 tests):** removing the realized history, or the time of day, raises the loss. One-sided Diebold-Mariano
    (DM) test, as in P106.
  * **F2 (12 tests):** the option-implied, VIX and calendar groups add nothing. This uses an **equivalence test**
    (two one-sided tests, TOST) with a margin of **1 point**. A row passes only if the 90% confidence interval of
    the group's contribution lies inside -1 to +1 point.
  * **F3 (8 tests):** two questions about the `mcsinfo` hybrid.
    * Does it beat mcsGARCH? One-sided DM test.
    * Is it **non-inferior** to the full hybrid? Its loss must be at most 1 point above the full hybrid's (a
      one-sided test against the margin).
* **A row passes** only if both of these hold:
  * its Holm-adjusted p is below 0.05;
  * all five seed sets point the same way. For example, for an equivalence row, all five seed sets' contributions
    must lie inside -1 to +1.
* **Why 1 point.**
  * P106's own plan used "more than 1 point" as the size that would contradict "the option-implied group adds
    nothing".
  * It is far above seed noise (about 0.1 point).
  * It is 16% to 26% of the hybrid's gain on these indices.
  * Its weakness, stated in the plan: Bank Nifty's time of day, one of P106's two "carrier" groups, was worth only
    about 0.9 point with seed set 7.
  * So the plan also repeats every margin test at **0.5 point** (S3) and reports, for every row, the smallest
    margin that would pass.
* **Secondary families.**
  * S1: the three groups "add something" (one-sided DM).
  * S2: the full hybrid beats `mcsinfo`.
  * S3: the 0.5-point margin.
  * S4a and S4b: pooled over three and over two indices, with Driscoll-Kraay errors that let the indices move
    together.
  * S5: the same equivalence tests on Nifty's committed losses.
* **Robustness.**
  * R1: seed set 7 alone.
  * R2: all five seed sets.
  * R3: a delta-method variance.
  * R4: without 4 and 5 June 2024.
  * R5: without the 1% of days with the largest mcsGARCH loss.
  * R6: one Holm adjustment over all 28 primary rows.
* **Placebo.** PL1 and PL2 compare the full hybrid with itself under another seed set.
* **Exploratory.** E1: the November 2024 period change of what `mcsinfo` leaves out.
* **Claim-level rules** (Section 5 of the plan) turn these rows into a verdict per claim and per index.
  * Claim 2 is "qualified" if a group is equivalent within 1 point but S1 also finds it significantly helps.
  * Claim 3 "partly survives" if `mcsinfo` beats mcsGARCH but non-inferiority is not shown at some horizon.

## 3. The 28 primary rows (REAL DATA)

From `verdict.csv` and `equivalence_bounds.csv`. How to read each row type:

* **"removing X"**: the contribution of group X in points. Positive means the group helps.
* **"X adds nothing"**: the same contribution, with its 90% interval. The row passes if the interval lies inside -1
  to +1.
* **"`mcsinfo` beats mcsGARCH"**: the `mcsinfo` hybrid's loss minus mcsGARCH's, in points. Negative means `mcsinfo`
  is better.
* **"`mcsinfo` non-inferior"**: the `mcsinfo` hybrid's loss minus the full hybrid's, in points. Positive means the
  full hybrid is better. The upper end of its 90% interval must be below 1.

The Holm p is adjusted within the row's family (F1, F2 or F3). "Seed range" is the estimate in each of the five seed
sets (7, 101, 102, 103, 104) separately. An "(R6: ...)" note gives the verdict under one Holm adjustment over all 28
rows.

| Index | Horizon | Row | Estimate (points) | 90% interval or bound | p | Holm p | Seed sets agreeing | Seed range | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Bank Nifty | 15 min | removing realized | +1.79 |  | below 0.001 | below 0.001 | 5 of 5 | +1.58 to +1.92 | supported |
| Bank Nifty | 15 min | removing time of day | +1.28 |  | below 0.001 | below 0.001 | 5 of 5 | +0.88 to +1.44 | supported |
| Bank Nifty | 15 min | implied adds nothing | -0.22 | [-0.67, +0.24] | 0.002 | 0.007 | 5 of 5 | -0.32 to -0.04 | equivalent |
| Bank Nifty | 15 min | VIX adds nothing | +0.08 | [-0.01, +0.17] | below 0.001 | below 0.001 | 5 of 5 | -0.19 to +0.35 | equivalent |
| Bank Nifty | 15 min | calendar adds nothing | +0.10 | [-0.05, +0.25] | below 0.001 | below 0.001 | 5 of 5 | -0.02 to +0.15 | equivalent |
| Bank Nifty | 15 min | `mcsinfo` beats mcsGARCH | -4.00 |  | below 0.001 | below 0.001 | 5 of 5 | -4.03 to -3.94 | supported |
| Bank Nifty | 15 min | `mcsinfo` non-inferior | +0.13 | upper +0.57 | below 0.001 | 0.002 | 5 of 5 | -0.02 to +0.20 | non-inferior |
| Bank Nifty | 30 min | removing realized | +2.16 |  | below 0.001 | below 0.001 | 5 of 5 | +1.79 to +2.37 | supported |
| Bank Nifty | 30 min | removing time of day | +1.16 |  | below 0.001 | below 0.001 | 5 of 5 | +0.95 to +1.27 | supported |
| Bank Nifty | 30 min | implied adds nothing | -0.20 | [-0.82, +0.42] | 0.018 | 0.035 | 5 of 5 | -0.48 to +0.10 | equivalent (R6: not shown) |
| Bank Nifty | 30 min | VIX adds nothing | +0.14 | [+0.01, +0.27] | below 0.001 | below 0.001 | 5 of 5 | -0.24 to +0.17 | equivalent |
| Bank Nifty | 30 min | calendar adds nothing | +0.27 | [+0.10, +0.44] | below 0.001 | below 0.001 | 5 of 5 | +0.06 to +0.39 | equivalent |
| Bank Nifty | 30 min | `mcsinfo` beats mcsGARCH | -5.09 |  | below 0.001 | below 0.001 | 5 of 5 | -5.13 to -5.00 | supported |
| Bank Nifty | 30 min | `mcsinfo` non-inferior | +0.23 | upper +0.83 | 0.018 | 0.055 | 5 of 5 | +0.03 to +0.45 | not shown |
| Fin Nifty | 15 min | removing realized | +1.73 |  | below 0.001 | below 0.001 | 5 of 5 | +1.64 to +1.94 | supported |
| Fin Nifty | 15 min | removing time of day | +1.47 |  | below 0.001 | below 0.001 | 5 of 5 | +1.38 to +1.62 | supported |
| Fin Nifty | 15 min | implied adds nothing | +0.09 | [-0.35, +0.52] | below 0.001 | 0.001 | 5 of 5 | +0.05 to +0.20 | equivalent |
| Fin Nifty | 15 min | VIX adds nothing | +0.05 | [-0.04, +0.13] | below 0.001 | below 0.001 | 5 of 5 | -0.10 to +0.15 | equivalent |
| Fin Nifty | 15 min | calendar adds nothing | +0.14 | [+0.05, +0.23] | below 0.001 | below 0.001 | 5 of 5 | +0.04 to +0.21 | equivalent |
| Fin Nifty | 15 min | `mcsinfo` beats mcsGARCH | -4.17 |  | below 0.001 | below 0.001 | 5 of 5 | -4.23 to -4.12 | supported |
| Fin Nifty | 15 min | `mcsinfo` non-inferior | +0.65 | upper +1.17 | 0.136 | 0.272 | 5 of 5 | +0.56 to +0.76 | not shown |
| Fin Nifty | 30 min | removing realized | +2.05 |  | below 0.001 | below 0.001 | 5 of 5 | +1.86 to +2.19 | supported |
| Fin Nifty | 30 min | removing time of day | +1.50 |  | below 0.001 | below 0.001 | 5 of 5 | +1.29 to +1.78 | supported |
| Fin Nifty | 30 min | implied adds nothing | -0.07 | [-0.83, +0.69] | 0.023 | 0.035 | 5 of 5 | -0.18 to +0.17 | equivalent (R6: not shown) |
| Fin Nifty | 30 min | VIX adds nothing | -0.12 | [-0.25, +0.02] | below 0.001 | below 0.001 | 5 of 5 | -0.22 to +0.09 | equivalent |
| Fin Nifty | 30 min | calendar adds nothing | +0.14 | [-0.03, +0.31] | below 0.001 | below 0.001 | 5 of 5 | -0.01 to +0.29 | equivalent |
| Fin Nifty | 30 min | `mcsinfo` beats mcsGARCH | -5.53 |  | below 0.001 | below 0.001 | 5 of 5 | -5.62 to -5.41 | supported |
| Fin Nifty | 30 min | `mcsinfo` non-inferior | +0.50 | upper +1.33 | 0.161 | 0.272 | 5 of 5 | +0.42 to +0.68 | not shown |

In short: 25 of the 28 rows pass. The three that do not are all non-inferiority rows of `mcsinfo` (Bank Nifty at 30
minutes, Fin Nifty at both horizons). In every one of the three, all five seed sets still agree that `mcsinfo` is
less than 1 point worse. The test fails because the 90% interval is wide, not because the seed sets disagree. No
row failed the seed rule.

For scale, the full hybrid's gain over mcsGARCH (REAL DATA, `contributions.csv`, new-seed average) is 4.13 points
(15 minutes) and 5.32 points (30 minutes) on Bank Nifty, and 4.83 and 6.03 points on Fin Nifty.

## 4. Claim 2 in detail (REAL DATA)

### 4.1 How close to zero are the three groups?

From `equivalence_bounds.csv` and `tests.csv` (F2 and S3 rows).

* "Smallest passing margin" is the larger end of the 90% interval in absolute value: the margin at which the
  equivalence test would just pass.
* "As share of gain" divides that margin by the hybrid's gain. It is the largest share of the gain the group could
  carry and still be within the data's reach.
* The last two columns are the Holm-adjusted p of the equivalence test at 1 point (F2) and at 0.5 point (S3).

| Index, horizon | Group | Estimate | SE | 90% interval | Smallest passing margin | As share of gain | Holm p, 1 point | Holm p, 0.5 point |
|---|---|---|---|---|---|---|---|---|
| Bank Nifty, 15 min | implied | -0.22 | 0.28 | [-0.67, +0.24] | 0.67 | 16% | 0.007 | 0.921 |
| Bank Nifty, 15 min | VIX | +0.08 | 0.06 | [-0.01, +0.17] | 0.17 | 4% | below 0.001 | below 0.001 |
| Bank Nifty, 15 min | calendar | +0.10 | 0.09 | [-0.05, +0.25] | 0.25 | 6% | below 0.001 | below 0.001 |
| Bank Nifty, 30 min | implied | -0.20 | 0.38 | [-0.82, +0.42] | 0.82 | 16% | 0.035 | 0.921 |
| Bank Nifty, 30 min | VIX | +0.14 | 0.08 | [+0.01, +0.27] | 0.27 | 5% | below 0.001 | below 0.001 |
| Bank Nifty, 30 min | calendar | +0.27 | 0.10 | [+0.10, +0.44] | 0.44 | 8% | below 0.001 | 0.110 |
| Fin Nifty, 15 min | implied | +0.09 | 0.27 | [-0.35, +0.52] | 0.52 | 11% | 0.001 | 0.483 |
| Fin Nifty, 15 min | VIX | +0.05 | 0.05 | [-0.04, +0.13] | 0.13 | 3% | below 0.001 | below 0.001 |
| Fin Nifty, 15 min | calendar | +0.14 | 0.05 | [+0.05, +0.23] | 0.23 | 5% | below 0.001 | below 0.001 |
| Fin Nifty, 30 min | implied | -0.07 | 0.46 | [-0.83, +0.69] | 0.83 | 14% | 0.035 | 0.921 |
| Fin Nifty, 30 min | VIX | -0.12 | 0.08 | [-0.25, +0.02] | 0.25 | 4% | below 0.001 | below 0.001 |
| Fin Nifty, 30 min | calendar | +0.14 | 0.10 | [-0.03, +0.31] | 0.31 | 5% | below 0.001 | 0.003 |

What this says:

* **VIX** is pinned close to zero everywhere: its interval never reaches 0.3 point, and it passes even at 0.5
  point.
* **The calendar group** passes at 0.5 point except on Bank Nifty at 30 minutes (Holm 0.110). There its interval
  lies wholly above zero, from +0.10 to +0.44.
* **The option-implied group** is the least precise. Its intervals reach 0.52 to 0.83 points, so it cannot pass at
  0.5 point anywhere. The data rule out that it carries more than 11% to 16% of the gain. They do not rule out
  that it carries a few percent of the gain in either direction.
* **Four seed sets barely narrowed the intervals.** The standard errors of the option-implied group (0.28, 0.38, 0.27
  and 0.46) are close to those the plan derived from seed set 7 alone (0.26, 0.40, 0.27 and 0.43, plan Section 7).
  The width comes from which days happened, not from LightGBM's randomness.

### 4.2 Does any of the three groups add something? (S1)

From `verdict.csv` and `tests.csv`, S1 rows: a one-sided DM test that removing the group raises the loss, with Holm
over the 12 rows. "Within seed noise" means the estimate is smaller than twice the spread (SD) of the five seed sets'
estimates.

| Index | Horizon | Group | Estimate | p | Holm p | Seed sets with estimate above 0 | Seed SD | Within seed noise |
|---|---|---|---|---|---|---|---|---|
| Bank Nifty | 15 min | implied | -0.22 | 0.783 | 1.000 | 0 of 5 | 0.11 | no |
| Bank Nifty | 15 min | VIX | +0.08 | 0.073 | 0.658 | 2 of 5 | 0.22 | yes |
| Bank Nifty | 15 min | calendar | +0.10 | 0.129 | 0.904 | 4 of 5 | 0.08 | yes |
| Bank Nifty | 30 min | implied | -0.20 | 0.706 | 1.000 | 1 of 5 | 0.22 | yes |
| Bank Nifty | 30 min | VIX | +0.14 | 0.042 | 0.422 | 4 of 5 | 0.17 | yes |
| Bank Nifty | 30 min | **calendar** | **+0.27** | 0.004 | **0.046** | **5 of 5** | 0.14 | yes |
| Fin Nifty | 15 min | implied | +0.09 | 0.374 | 1.000 | 5 of 5 | 0.06 | yes |
| Fin Nifty | 15 min | VIX | +0.05 | 0.179 | 1.000 | 4 of 5 | 0.10 | yes |
| Fin Nifty | 15 min | **calendar** | **+0.14** | 0.004 | **0.046** | **5 of 5** | 0.07 | no |
| Fin Nifty | 30 min | implied | -0.07 | 0.563 | 1.000 | 2 of 5 | 0.15 | yes |
| Fin Nifty | 30 min | VIX | -0.12 | 0.925 | 1.000 | 1 of 5 | 0.11 | yes |
| Fin Nifty | 30 min | calendar | +0.14 | 0.093 | 0.745 | 4 of 5 | 0.11 | yes |

Two rows are significant under the plan's rule (Holm below 0.05 and all five seed sets above zero): the calendar
group on Bank Nifty at 30 minutes and on Fin Nifty at 15 minutes. Both are also "equivalent" in F2, so the plan's
label is "adds a little (under the margin)". Some points to note:

* **Both rows only just pass.** Their Holm p is 0.046.
* **The Fin Nifty row was expected**, because P106's seed set 7 already found it (Holm 0.035, P106 `tests.csv`).
* **The Bank Nifty row was not expected.** It is also flagged "within seed noise": its seed-set estimates run from
  +0.06 to +0.39.
* **Nifty goes the other way.** P106 found that removing the calendar group made the hybrid slightly better there.
  The sign of this small effect differs between indices.
* **The option-implied group and VIX add nothing significant anywhere.** The option-implied group's point estimate
  is negative (removing it *helps*) on Bank Nifty at both horizons and on Fin Nifty at 30 minutes.

### 4.3 The option-implied group at 30 minutes is the weakest pass

From `tests.csv`, rows `F2_*_next30_implied` and their R1 to R6 copies. Each robustness family has its own Holm
adjustment over its rows (28, or 16 for R3), so the Holm p values differ from the primary family's.

| Sample | Bank Nifty: estimate, p, Holm p | Fin Nifty: estimate, p, Holm p |
|---|---|---|
| Primary (new-seed average, Holm over 12) | -0.20, 0.018, **0.035** | -0.07, 0.023, **0.035** |
| R1: seed set 7 alone | -0.48, 0.094, 0.282 | +0.17, 0.028, 0.111 |
| R2: all five seed sets | -0.26, 0.025, 0.074 | -0.03, 0.016, 0.070 |
| R3: delta-method variance | -0.20, 0.017, 0.086 | -0.07, 0.023, 0.086 |
| R4: without 4 and 5 June 2024 | -0.24, 0.022, 0.086 | +0.08, 0.022, 0.086 |
| R5: without the top 1% loss days | -0.05, 0.005, **0.026** | -0.13, 0.017, 0.069 |
| R6: one Holm over all 28 primary rows | -0.20, 0.018, 0.088 | -0.07, 0.023, 0.088 |

The raw p stays between 0.005 and 0.028 in every sample except seed set 7 alone on Bank Nifty (0.094). The Holm p
depends on how many rows share the adjustment. The plan had warned that this row was the one most likely to stay
"not shown" (plan Section 7). It passed in the primary family but not under most stricter readings. **A fair
summary is: the option-implied group's value at 30 minutes is within about 0.8 point of zero, but the evidence for
"within 1 point" is borderline.**

## 5. Claim 3 in detail: the `mcsinfo` hybrid (REAL DATA)

From `contributions.csv`, `verdict.csv`, `equivalence_bounds.csv`, `shares.csv` and `tests.csv`. The columns are:

* **Gains:** % below mcsGARCH's loss.
* **N:** the `mcsinfo` hybrid's loss minus the full hybrid's, in points. Positive means the full hybrid is better.
* **Upper bound:** the upper end of N's 90% interval.
* **Share of gain lost:** N as a share of the full hybrid's gain, with its 95% stationary-bootstrap interval.

| Index, horizon | Full hybrid's gain | `mcsinfo`'s gain | N | Upper bound | Non-inferior within 1 point? (F3 b, Holm p) | Full beats `mcsinfo`? (S2: p, Holm p) | Seed sets with N above 0 | Share of gain lost |
|---|---|---|---|---|---|---|---|---|
| Bank Nifty, 15 min | 4.13% | 4.00% | +0.13 | 0.57 | **yes** (0.002) | no (0.312, 0.534) | 4 of 5 | 3.2% (-10.4% to 15.5%) |
| Bank Nifty, 30 min | 5.32% | 5.09% | +0.23 | 0.83 | not shown (0.055) | no (0.267, 0.534) | 5 of 5 | 4.3% (-10.7% to 18.0%) |
| Fin Nifty, 15 min | 4.83% | 4.17% | +0.65 | 1.17 | not shown (0.272) | no (0.019, 0.076) | 5 of 5 | 13.6% (0.9% to 26.3%) |
| Fin Nifty, 30 min | 6.03% | 5.53% | +0.50 | 1.33 | not shown (0.272) | no (0.160, 0.481) | 5 of 5 | 8.3% (-7.3% to 24.0%) |

* **On Bank Nifty the `mcsinfo` hybrid is close to the full hybrid** (0.13 and 0.23 points behind). Non-inferiority
  misses at 30 minutes by a little (Holm 0.055). It passes on seed set 7 alone (R1, Holm 0.032) but not with all
  five seed sets (R2, Holm 0.070).
* **On Fin Nifty it is behind by about half a point in every seed set.**
  * At 15 minutes the bootstrap interval of the share of the gain it loses lies above zero (0.9% to 26.3%).
  * The DM test of the same gap has a raw p of 0.019.
  * But the DM test does not pass its Holm adjustment (0.076).
  * Taken together: the information mcsGARCH lacks probably adds something on Fin Nifty at 15 minutes. How much is
    uncertain, possibly more than 1 point.
* **This differs from Nifty.** There the `mcsinfo` hybrid was *better* than the full hybrid. In the new-seed average
  of P106's committed Nifty losses it is 0.29 and 0.18 points better (S5 rows): its loss is 6.07% and 8.07% below
  mcsGARCH's, against 5.78% and 7.88% for the full hybrid.

Under the plan's rule claim 3 "partly survives" on both indices. `mcsinfo` beats mcsGARCH at both horizons, but
non-inferiority is not shown at 30 minutes on Bank Nifty or at either horizon on Fin Nifty. The full hybrid is not
significantly better anywhere, so the claim is not contradicted.

## 6. Pooled tests and the Nifty re-analysis (REAL DATA, secondary)

### 6.1 Pooled over indices (S4a, S4b)

From `pooled.csv` and `tests.csv`.

* Each index's daily series is in points of its own mcsGARCH loss.
* The equal-weight average uses P107's Driscoll-Kraay errors.
* Three-index pool: Nifty, Bank Nifty and Fin Nifty on their 604 common test days (to 27 January 2026). Nifty is the
  new-seed average of P106's committed losses.
* Two-index pool: Bank Nifty and Fin Nifty on their 604 common days.
* Holm is applied over the 14 rows of each pool.
* Estimates are in points. Equivalence rows show the 90% interval, and every row shows its Holm p.

| Row | Three indices, 15 min | Three indices, 30 min | Bank + Fin Nifty, 15 min | Bank + Fin Nifty, 30 min |
|---|---|---|---|---|
| removing realized | +1.79 (below 0.001) | +2.38 (below 0.001) | +1.75 (below 0.001) | +2.13 (below 0.001) |
| removing time of day | +1.41 (below 0.001) | +1.50 (below 0.001) | +1.35 (below 0.001) | +1.33 (below 0.001) |
| implied adds nothing | -0.02 [-0.31, +0.26] (below 0.001) | -0.05 [-0.52, +0.42] (below 0.001) | -0.01 [-0.35, +0.33] (below 0.001) | -0.10 [-0.61, +0.40] (0.005) |
| VIX adds nothing | +0.06 [+0.00, +0.13] (below 0.001) | -0.01 [-0.09, +0.08] (below 0.001) | +0.07 [+0.01, +0.13] (below 0.001) | +0.03 [-0.07, +0.13] (below 0.001) |
| calendar adds nothing | +0.10 [+0.01, +0.19] (below 0.001) | +0.16 [+0.02, +0.30] (below 0.001) | +0.12 [+0.03, +0.21] (below 0.001) | +0.20 [+0.06, +0.34] (below 0.001) |
| `mcsinfo` beats mcsGARCH | -4.59 (below 0.001) | -5.79 (below 0.001) | -4.22 (below 0.001) | -5.37 (below 0.001) |
| `mcsinfo` non-inferior | +0.24 (below 0.001) | +0.32 (0.012) | +0.42 (0.011) | +0.43 (0.045) |

* All 28 pooled rows pass.
* Pooling narrows the intervals. The option-implied group's pooled interval at 15 minutes ends 0.31 points from
  zero, for example.
* **The calendar group's pooled 90% interval lies above zero at both horizons in both pools.** This is descriptive:
  no pooled "adds something" test was registered. On these 604 days the calendar group's contribution is positive
  even on Nifty (+0.05 and +0.08 points, `pooled.csv`, column `estimate_nifty`). That is unlike Nifty's full 792
  days (Section 6.2).
* **Pooled non-inferiority of `mcsinfo` passes.** But in the two-index pool it passes only narrowly at 30 minutes
  (Holm 0.045). An average over indices can hide that one index (Fin Nifty) does not pass on its own.

### 6.2 The same equivalence tests on Nifty (S5)

From `equivalence_bounds.csv` and `tests.csv`, rows `S5_*`. The data are P106's committed Nifty daily losses (792
days), averaged over seed sets 101 to 104. Holm is applied over 8 rows. These data and P106's tests on them were
already public, so this is a re-analysis with a new test, not new evidence.

| Row | 15 min: estimate, 90% interval or bound, Holm p | 30 min: estimate, 90% interval or bound, Holm p |
|---|---|---|
| implied adds nothing | +0.12, [-0.50, +0.74], 0.010 | -0.08, [-0.64, +0.48], 0.007 |
| VIX adds nothing | +0.03, [-0.08, +0.14], below 0.001 | -0.09, [-0.20, +0.03], below 0.001 |
| calendar adds nothing | -0.24, [-0.62, +0.14], 0.002 | -0.15, [-0.52, +0.21], below 0.001 |
| `mcsinfo` non-inferior | -0.29, upper +0.18, below 0.001 | -0.18, upper +0.43, 0.002 |

All 8 pass. On Nifty, the equivalence test now supports P106's wording that the three groups add nothing (within
1 point), and the `mcsinfo` hybrid is non-inferior to the full hybrid.

## 7. Robustness and placebo (REAL DATA)

### 7.1 What changes under the robustness checks

From `tests.csv` (R1 to R6 rows). For each robustness family, these are the primary rows whose pass or fail (Holm p
below 0.05) differs from the primary result:

| Check | Rows | Rows that change |
|---|---|---|
| R1: seed set 7 alone | 28 | implied at 30 min fails on both indices; Bank Nifty `mcsinfo` non-inferior at 30 min passes (Holm 0.032) |
| R2: all five seed sets | 28 | implied at 30 min fails on both indices (Holm 0.074, 0.070) |
| R3: delta-method variance | 16 | implied at 30 min fails on both indices (Holm 0.086, 0.086) |
| R4: without 4 and 5 June 2024 | 28 | implied at 30 min fails on both indices (Holm 0.086, 0.086) |
| R5: without the top 1% loss days | 28 | implied at 30 min fails on Fin Nifty (Holm 0.069) |
| R6: one Holm over all 28 | 28 | implied at 30 min fails on both indices (Holm 0.088, 0.088) |

Nothing else changes:

* every F1 row and every "`mcsinfo` beats mcsGARCH" row passes in every check;
* every VIX and calendar equivalence row passes in every check;
* the Fin Nifty non-inferiority rows fail in every check.

Dropping the general-election result day and the day after (R4) leaves every estimate close to the primary one.
For example, removing the realized history costs 1.88 instead of 1.79 points on Bank Nifty at 15 minutes.

### 7.2 Placebo: the same hybrid with different seeds

From `tests.csv`, PL1 and PL2 rows. Each row compares the full hybrid with seed set 101, 102, 103 or 104 against
the same hybrid with seed set 7. Only the seeds differ.

* **PL1 (equivalence within 1 point):** all 16 rows pass (Holm p 0.001 or below). The equivalence test does say
  "nothing" when nothing but the seeds differs.
* **PL2 (one-sided DM, "seed set 7 is better"):** none of the 16 rows is significant (Holm p 0.354 to 1.000).
* **Seed set 7 on Bank Nifty.** Every new seed set beat seed set 7 there, by 0.17 to 0.51 points. Their DM
  statistics, -1.28 to -3.23, point the other way from the PL2 test, so its p is near 1. The plan registered no
  test in this direction, so no p-value is claimed for it.
  * It matches Section 1: seed set 7 had the smallest Bank Nifty gain of the five.
  * The committed headline numbers for Bank Nifty (3.92 and 5.03 points) are therefore on the low side of what the
    same model gives with other seeds (4.09 to 4.21 and 5.23 to 5.55).
  * On Fin Nifty it was the other way round, by less: every new seed set was a little worse than seed set 7, by
    0.002 to 0.189 points at 15 minutes and 0.037 to 0.104 points at 30 minutes (PL1 estimates, which are the new
    seed set's loss minus seed set 7's, in points).

## 8. Exploratory and post hoc results (REAL DATA, no adjustment)

### 8.1 E1: did the value of what mcsGARCH lacks change in November 2024?

From `tests.csv` (E1 rows) and `periods.csv`. N is the `mcsinfo` hybrid's loss minus the full hybrid's (positive
means the information mcsGARCH lacks helps). It is compared between test days to 19 November 2024 and from 20
November 2024. Two-sided test, no adjustment.

| Index, horizon | N before | N after | Change | p |
|---|---|---|---|---|
| Bank Nifty, 15 min | +0.31 | -0.06 | -0.37 | 0.478 |
| Bank Nifty, 30 min | +0.72 | -0.31 | -1.03 | 0.157 |
| Fin Nifty, 15 min | -0.04 | +1.55 | +1.59 | 0.009 |
| Fin Nifty, 30 min | -0.14 | +1.37 | +1.51 | 0.116 |

* On Bank Nifty the direction matches P106's Nifty finding (worth more before the cut), but it is not significant.
* On Fin Nifty it goes the other way: the information mcsGARCH lacks was worth more *after* the cut (significant
  at 15 minutes, unadjusted).
* The plan stated no direction for these indices.
* P106's placebo showed that a seed-only period change can reach small p-values, so a single unadjusted p of 0.009
  is weak evidence.

### 8.2 Post hoc: the p-values of the period table

The registered descriptive table `periods.csv` shows, for each index and horizon, the period values of each
contribution and of two other quantities, each with a two-sided p-value:

* G, the hybrid's gain over mcsGARCH;
* B, the information-matched benchmark `mcsgarch_iv`'s extra loss over mcsGARCH.

Every p-value of the study belongs in `tests.csv`, so these 28 were added after the first analysis as **post hoc**
rows (`PH_*`, no adjustment; see Deviations). With 28 unadjusted tests, about 1.4 p-values below 0.05 are expected
by chance alone (28 x 0.05); five are below 0.05.

| Index, horizon | Quantity | Before 20 Nov 2024 | From 20 Nov 2024 | Change | p (post hoc) |
|---|---|---|---|---|---|
| Bank Nifty, 15 min | option-implied contribution | +0.47 | -0.95 | -1.42 | 0.008 |
| Bank Nifty, 30 min | option-implied contribution | +0.80 | -1.31 | -2.11 | 0.004 |
| Fin Nifty, 15 min | option-implied contribution | -0.21 | +0.47 | +0.68 | 0.187 |
| Fin Nifty, 30 min | option-implied contribution | -0.48 | +0.48 | +0.96 | 0.270 |
| Bank Nifty, 15 min | G, hybrid's gain | 3.48 | 4.84 | +1.36 | 0.216 |
| Bank Nifty, 30 min | G, hybrid's gain | 3.70 | 7.11 | +3.40 | 0.019 |
| Fin Nifty, 15 min | G, hybrid's gain | 3.78 | 6.17 | +2.39 | 0.061 |
| Fin Nifty, 30 min | G, hybrid's gain | 4.48 | 8.12 | +3.64 | 0.016 |
| Bank Nifty, 15 min | B, `mcsgarch_iv` minus mcsGARCH | -2.65 | -2.97 | -0.32 | 0.873 |
| Bank Nifty, 30 min | B, `mcsgarch_iv` minus mcsGARCH | -3.30 | -4.83 | -1.53 | 0.632 |
| Fin Nifty, 15 min | B, `mcsgarch_iv` minus mcsGARCH | -2.66 | -2.39 | +0.27 | 0.898 |
| Fin Nifty, 30 min | B, `mcsgarch_iv` minus mcsGARCH | -2.57 | -2.98 | -0.41 | 0.893 |

The fifth p-value below 0.05 is the VIX contribution on Fin Nifty at 30 minutes (from -0.29 to +0.12 points, p
0.009). The other contributions (VIX elsewhere, time of day, calendar, realized history) have p between 0.059 and
0.981 (`tests.csv`, `PH_*`). What these suggest, with the warning that they are post hoc, unadjusted and ignore
seed noise:

* **The option-implied group's value fell after the cut on Bank Nifty.** It was positive before and negative
  after, at both horizons. This is the same direction as P106's Nifty result at 30 minutes (+1.14 to -1.16 points,
  Holm 0.051 there). On Fin Nifty it moved the other way, not significantly.
* **On these two indices the hybrid's own gain grew after the cut at 30 minutes, and the benchmark did not move.**
  * `mcsgarch_iv` was better than mcsGARCH in both periods on both indices (B negative), and B did not change
    significantly (p 0.632 to 0.898).
  * This is not the pattern P106 found on Nifty, where `mcsgarch_iv` lost its advantage after the cut and the
    hybrid's gain was steady.
  * P106's Part 2 wording ("the benchmark lost its edge, not the hybrid got better") is a Nifty result. It should
    not be carried over to Bank Nifty or Fin Nifty without a registered test.

### 8.3 Shares of the gain and the ranking of the groups (descriptive)

From `shares.csv`: each group's contribution as a share of the hybrid's gain (new-seed average), with the 95%
stationary-bootstrap interval (10,000 draws, mean block 10 days).

| Group | Bank Nifty, 15 min | Bank Nifty, 30 min | Fin Nifty, 15 min | Fin Nifty, 30 min |
|---|---|---|---|---|
| realized history | 43.3% (30.0% to 58.7%) | 40.6% (28.7% to 52.7%) | 35.8% (22.5% to 51.6%) | 34.1% (19.9% to 51.2%) |
| time of day | 31.1% (21.6% to 42.6%) | 21.8% (15.3% to 29.5%) | 30.5% (20.8% to 42.9%) | 24.8% (14.0% to 35.8%) |
| option-implied | -5.2% (-20.6% to 8.6%) | -3.8% (-19.8% to 11.6%) | 1.8% (-9.8% to 12.2%) | -1.2% (-16.8% to 13.9%) |
| VIX | 2.0% (-0.8% to 5.0%) | 2.6% (-0.5% to 5.5%) | 1.0% (-1.0% to 3.3%) | -1.9% (-5.1% to 0.9%) |
| calendar | 2.5% (-1.8% to 7.0%) | 5.1% (1.4% to 9.1%) | 3.0% (0.5% to 6.3%) | 2.3% (-1.6% to 6.1%) |
| `mcsinfo` (everything mcsGARCH lacks) | 3.2% (-10.4% to 15.5%) | 4.3% (-10.7% to 18.0%) | 13.6% (0.9% to 26.3%) | 8.3% (-7.3% to 24.0%) |

* The realized history and the time of day come first and second on every index and horizon, Nifty included
  (`other_index_ranking.csv`).
* The order of the three small groups differs between indices.
* The rank correlation of the five contributions with Nifty's is 0.6 on Bank Nifty and 0.7 on Fin Nifty, at both
  horizons.

## 9. What we expected, and what happened

| Row | Registered expectation (plan Section 6) | Outcome |
|---|---|---|
| F1 realized, time of day | supported on both indices at both horizons | **as expected**: all eight supported, Holm below 0.001 |
| F2 VIX and calendar | equivalent everywhere | **as expected**: all eight equivalent |
| F2 implied, 15 minutes | equivalent on both indices | **as expected** (Holm 0.007 and 0.001) |
| F2 implied, 30 minutes | may not be shown, at least on Bank Nifty | **better than expected but fragile**: equivalent on both (Holm 0.035), fails under R6 and in most robustness samples |
| S1 | Fin Nifty calendar at 15 minutes may be significant; nothing else | **partly**: Fin Nifty calendar at 15 min significant as expected; **also** Bank Nifty calendar at 30 min (not expected) |
| F3 (a) | supported everywhere | **as expected** |
| F3 (b), S2 | non-inferior; S2 not significant | **partly**: non-inferior only on Bank Nifty at 15 minutes; S2 not significant anywhere, as expected |
| S5 (Nifty) | all equivalent and non-inferior at 1 point | **as expected** |
| PL1, PL2 | PL1 all pass, PL2 none | **as expected**; on Bank Nifty all new seed sets were better than seed set 7 (no test in that direction) |

## 10. Checks

From `reproduction_check.csv` and `run_info.json`.

* **Reproduction of the committed forecasts.** The full hybrid with seed set 7 is identical to the last bit to the
  committed `hybrid_base` forecasts, together with `y`, `mcsgarch` and `mcsgarch_iv`, on every row:
  * Bank Nifty: 16,500 rows at 15 minutes and 15,840 at 30 minutes (`results/pipeline_nse_banknifty/`);
  * Fin Nifty: 15,100 and 14,496 rows (`results/pipeline_nse_finnifty/`).
* **Reproduction of P106's work files.** The 24 seed-7 files of the five group variants and the full hybrid (two
  indices, two horizons) are identical to P106's git-ignored work files on every row, with a largest relative
  difference of 0. Dates, minutes, `y`, `mcsgarch` and `mcsgarch_iv` are also identical. P106's work files were
  read from a copy in `data/processed/p106_ref/` (see Deviations).
* **Reproduction of P106's numbers.** The 20 seed-7 contributions and DM statistics recomputed here equal P106's
  committed `tests.csv` rows `S5_*` and `S6_*` (largest absolute difference 2.2e-16, rounding only).
* **Same benchmarks in every seed set.** mcsGARCH's and `mcsgarch_iv`'s daily QLIKE are identical in all five seed
  sets on every day (660 Bank Nifty days, 604 Fin Nifty days, both horizons). The script also refuses to run if `y`,
  `mcsgarch` or `mcsgarch_iv` differ between any two work files of an index and horizon.
* **Inputs.** Every raw file each run read matches `data/MANIFEST.csv`: 1,891 files for Bank Nifty and 1,711 for
  Fin Nifty, none differing (`run_info.json`, `input_sha256`).
* **Post hoc rows.** After the post hoc rows were added (Deviations, item 1), the analysis was run again. Its output
  was compared file by file with the first analysis using `cmp`:
  * every output file except `tests.csv`, `tests_detail.csv`, `run_info.json` and `analyze_stdout.txt` is byte for
    byte the same;
  * the 288 rows of the first `tests.csv` are unchanged.
* **Tests.** `tests/test_p110_gain_stress_test.py` (SYNTHETIC only) has 12 tests. They cover:
  * TOST equals the two shifted DM tests and the 90% interval rule;
  * its size at the margin and its power on simulated data;
  * the delta-method variant;
  * the seed rule and the claim-level rules;
  * the pooled test;
  * an end-to-end dry run on three simulated markets.

  The full test suite passes (run before the phase-2 commit; see the commit message).

## 11. Run time

From `run_info.json` (`stage1_wall_clock`, `seconds_stage2`) and `timing.csv`.

* **Forecasts (stage 1)** ran on one process:
  * Bank Nifty: 08:35:01 to 09:54:35 UTC, 79.6 minutes;
  * Fin Nifty: 09:55:35 to 11:06:05 UTC, 70.5 minutes;
  * 10 jobs per index, each fitting 7 variants, took 370.0 to 535.2 seconds per job;
  * the jobs' own times add up to 8,834.7 seconds (4,685.8 for Bank Nifty, 4,148.9 for Fin Nifty).
* **Analysis (stage 2):** 18.1 seconds.
* The plan expected 1 to 1.75 hours per index and 2 to 3.5 hours in all. The forecasts took about 2.5 hours.

## 12. Deviations from the plan

1. **28 post hoc rows were added after the first analysis (outcomes had been seen).**
   * **Why.** The registered descriptive table `periods.csv` reports a two-sided p-value (P106's
     `period_diff_test`) for each quantity, but the plan says every p-value goes in `tests.csv`.
   * **What was added.** `stage_analyze` now also writes those p-values to `tests.csv` as rows `PH_*` with role
     "post hoc" and no adjustment. There is one row per index, horizon and quantity: G, B and the five groups.
     The `mcsinfo` row was already registered as E1.
   * **Effect.** No registered number changed (Section 10).
2. **`analysis_commit` in `run_info.json` reads 5128e53** (the plan commit), because `analyze` records the commit
   that was checked out. The analysis ran on that commit plus the post hoc change above, which is in the phase-2
   commit.
3. **P106's `check` stage was also run once during the run**, at 09:54:54 UTC, after the Bank Nifty forecasts had
   finished and before Fin Nifty's started. It ran on this study's work folder
   (`experiments/p106_gain_decomposition.py check --indexes BANKNIFTY --work data/processed/p110`).
   * It compares the seed-7 full hybrid's forecasts with the committed ones and computes no loss.
   * All rows were identical (16,500 and 15,840).
   * The plan did not mention it. The same comparison is in `reproduction_check.csv`.
4. **Commands.** The plan's commands set `PYTHONPATH=src`; the run used the absolute path of the same `src` folder
   (a sandbox rule). The two `run` commands were started by a small shell wrapper that ran them one after the
   other, with their output going to the registered `run_stdout_*.txt` files. Same code and arguments.
5. **P106's work files were read from a copy.** P106's git-ignored seed-7 work files for Bank Nifty and Fin Nifty
   were in the P106 agent's own working copy. Before the plan was committed (08:22 UTC), they were copied into
   `data/processed/p106_ref/` (git-ignored), which is the script's default `--p106-work` folder. The plan's check
   applies "if those git-ignored files are still on disk"; they were, so the check was run.
6. **A labelling quirk in `reproduction_check.csv`.** In the 24 "vs P106 work file" rows the column `rows_matched`
   is a 0 or 1 flag (1 = the same number of rows), not a count. The count of identical rows is in
   `forecast_n_identical`, which equals `rows_rerun` in every row. It was left as registered.
7. **Nothing was dropped.** Both indices were run in full; no row is "not computed".

## 13. Caveats

1. **No new days.** Every test uses days P106 used, and P106's seed-7 results on them were public. The new seed sets
   remove one source of chance (LightGBM's randomness) but not the other (which days happened). Only data after
   these test periods can confirm the results.
2. **The margin is a choice.** One point is 16% to 26% of the gain on these indices (plan Section 3). It is not small
   next to Bank Nifty's time-of-day contribution (1.28 and 1.16 points here). The 0.5-point rows (Section 4.1)
   and the reported intervals let a reader apply another margin.
3. **Leave-one-group-out measures what a group adds given all the others.** A group can look useless because
   another stands in for it.
   * The `mcsinfo` variant covers this for the option market, VIX and calendar together.
   * It does not cover it between the realized history and the time of day.
   * The five group shares add up to less than the whole gain.
4. **The protocol is fixed.** Each variant keeps the headline's LightGBM settings. A tuned `mcsinfo` hybrid might
   do better or worse.
5. **Five seed sets** show whether a sign is stable. They cannot estimate seed noise precisely, and the DM, TOST and
   period tests do not include seed noise in their p-values.
6. **The three indices move together.** Bank Nifty and Fin Nifty share news, VIX and many constituents. P107 found
   the three indices are worth about two independent ones. So agreement between them is less than two independent
   confirmations, and the pooled tests average correlated evidence.
7. **The two "adds a little" calendar rows only just pass** (Holm 0.046). The Bank Nifty one is within seed noise.
   With a slightly different adjustment either could go away; claim 2 would then read "survives" on that index.
8. **The post hoc period results (Section 8.2) are hypotheses**, not findings. They are unadjusted, chosen after
   seeing the table, and ignore seed noise.

## 14. What this changes in the project's wording of P106

* **Claim 1 stands, now on three indices with five seed sets each.** "Recent realized volatility and the time of day
  carry the gain" holds on Bank Nifty and Fin Nifty with every seed set. There the realized history is worth 34%
  to 43% of the gain and the time of day 22% to 31%.
* **Claim 2 should be reworded.** Instead of "the option market, VIX and the calendar add nothing", write:
  * removing the option-implied inputs or VIX changes the loss by less than 1 point (1% of mcsGARCH's loss) on
    Nifty, Bank Nifty and Fin Nifty (equivalence tests);
  * for the option-implied group at 30 minutes this is borderline (it fails under stricter adjustments);
  * the calendar inputs help a little on Bank Nifty (30 minutes) and Fin Nifty (15 minutes), by about 0.1 to 0.3
    points, and do not help on Nifty.

  The sentence "any sentence that credits the option-implied features with the gain should go" still holds: no
  index shows a significant contribution from them.
* **Claim 3 should be reworded.** Instead of "a hybrid that sees only mcsGARCH's own information gets the whole
  gain", write:
  * it beats mcsGARCH on all three indices and gets nearly all of the gain on Nifty and Bank Nifty;
  * on Fin Nifty it gets most of the gain but trails the full hybrid by about half a point (8% to 14% of the
    gain), which is not significant but cannot be ruled out to exceed 1 point.

  "The gain does not need the option market" remains fair on all three indices.
* **The November 2024 wording stays a Nifty result.** On Bank Nifty and Fin Nifty a post hoc look suggests the
  pattern differs: the hybrid's own gain grew and `mcsgarch_iv` did not lose its edge. This needs its own registered
  test before anything is said about those indices.
* **The Bank Nifty headline is on the low side of seed noise.** No change to the registered headline numbers is
  proposed. When Bank Nifty's gain is quoted, it is fair to add that other seed sets give 0.2 to 0.5 points more.

## 15. Files

All in `results/p110_gain_stress_test/` (REAL DATA):

* `tests.csv`: every p-value, 316 rows (28 primary, 68 secondary, 156 robustness, 32 placebo, 4 exploratory, 28
  post hoc), with P106's columns. `tests_detail.csv` adds the identifiers and sample details.
* `verdict.csv`: the 28 primary rows and the 16 S1 and S2 rows, with seed agreement, seed SD and verdict (and the
  verdict under R6).
* `claims.csv`: the claim-level verdicts per index.
* `equivalence_bounds.csv`: estimates, standard errors, 90% intervals, smallest passing margins and TOST p-values
  at 1 and 0.5 point for every equivalence and non-inferiority row (F2, F3 (b), S5).
* `contributions.csv`, `seed_repeats.csv`, `shares.csv`, `periods.csv`, `pooled.csv`, `other_index_ranking.csv`:
  the tables above.
* `daily_losses_<index>_<target>_s<seed>.csv` (20 files): each variant's daily QLIKE, from which every number can be
  recomputed.
* `reproduction_check.csv`, `timing.csv`, `run_info.json`: checks, timings, plan commit, seeds and input hashes.
* `run_stdout_banknifty.txt`, `run_stdout_finnifty.txt`, `analyze_stdout.txt`: logs.

The forecasts of every variant are in `data/processed/p110/` (git-ignored). They can be remade with the plan's
commands in about 2.5 hours on one core.
