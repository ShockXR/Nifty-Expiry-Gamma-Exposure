# P114: does the fall in intraday persistence survive when mcsGARCH is fixed?

Study P114, run by an AI agent (Claude) working for Shreyan Thanki on 3 November 2026.
**All numbers below are REAL DATA** (Nifty and Bank Nifty one-minute index data, with the replication's data
steps and windows) unless marked SYNTHETIC. Every number comes from a file in
`results/p114_persistence_fixed_model/` (file named next to each table) or, where marked, from an earlier
committed results file.

* Registered plan: `docs/prereg/P114_PERSISTENCE_FIXED_MODEL.md`, commit `86e386635284b24d876c32a7aecc63a1ecd23886`
  (committed 10:11:25 UTC and pushed before any model with P109's fixes was estimated on real data, and before any
  bootstrap draw, test statistic or drift measure of this study was computed).
* Code (all in the plan commit, unchanged since): `experiments/p114_persistence_fixed_model.py`; new code after the
  existing code in `src/niftygex/volatility.py` (`ext_impulse_response`, `irf_half_life`, `equivalent_persistence`,
  `ext_persistence_summary`, `solve_beta_for_half_life`, `fit_garch_ext_fixed`). Tests (SYNTHETIC only):
  `tests/test_p114_persistence_fixed_model.py`. One **post hoc** script was added after the results were seen
  (`experiments/p114_posthoc_night_cap.py`, descriptive only, no p-values; Section 3).
* Every p-value of the study: `results/p114_persistence_fixed_model/tests.csv` (20 rows: 1 primary, 13 secondary,
  2 robustness, 4 exploratory).

Words used below.

* **Half-life**: how many trading minutes it takes for the extra volatility caused by one surprise move to fall to
  half of its first effect. Long half-life = shocks linger; short = they die out fast.
* **Reference model**: the replication's mcsGARCH, refitted exactly as in its Table 2.
* **Fixed model (F)**: the same model with three of P109's fixes: X1 (a daily level from a HAR model of daily
  realized variance), X2 (the minute-by-minute part fades toward its average overnight) and X3 (a term that feeds
  in today's average squared return so far).
* **Equivalent persistence** rho = 0.5 ** (1 / half-life): the alpha + beta that a plain GARCH(1,1) would need to
  have the same half-life. For the reference model it equals alpha + beta. **D** = rho after minus rho before;
  negative means shocks die out faster after the change (the paper's direction).
* **Kept share K**: the fall in log half-life under the fixed model divided by the fall under the reference. K = 1
  means the fixed model shows the same proportional fall; K = 0 means no fall at all.

## 1. The answer in plain words (REAL DATA)

The anchor paper (Chhimwal, Pandey and Pandey 2025) reports that after the expiry-day change, a burst of intraday
volatility dies out much faster. Our replication confirms this with the paper's own model: on the inferred before
window, the half-life of a shock falls from 359 to 72 minutes for Nifty and from 162 to 38 minutes for Bank Nifty
(`docs/REPLICATION_NSE.md`, finding 4). Study P109 then found that this model (mcsGARCH) has no slowly moving level
inside the day. Such a model can show long-lived shocks simply because the day's volatility level wanders and the
model has no other way to follow it. If the before window wandered more, part of the "faster decay" could be an
artefact. This study refitted the model with three of P109's fixes (the fixed model) on the same days, and asked
whether the fall survives.

**Does the fall survive? On the paper's window, for Nifty, yes. In the point estimate it is as large as before or
larger, but its size is very uncertain.**

* Under the fixed model, the Nifty half-life of a shock at 12:15 is **141 minutes before and 18 minutes after** the
  change (the reference model: 359 and 72).
* The change in equivalent persistence is **-0.0331, with a 95% bootstrap interval of -0.0579 to -0.0026**. 98% of
  the 400 bootstrap draws show a fall (one-sided bootstrap p 0.022). The registered verdict is **"survives"**.
* The fall in log half-life under the fixed model is 1.29 times the reference's (kept share K = 1.29), so the size
  label is **"keeps most of the fall"**. But the paired 95% interval of K runs from -0.04 to 6.91, and the
  difference between the two models' log falls is +0.46 with an interval of -1.42 to +1.79. So the data cannot
  tell whether the fixed model's fall is smaller or larger than the reference's; they only say that it is a fall.
* The secondary tests of the same comparison agree: likelihood-ratio test of equal half-life 16.65 (p 0.00004, Holm
  0.0001); alpha + beta falls from 0.9925 to 0.9552 (bootstrap p 0.022; likelihood-ratio p 0.00007).
* A post hoc check with a measure that has no "night cap" (the share of a shock's first effect still there at the
  close) agrees: 0.44 before, 0.15 after, change -0.30 with interval -0.74 to -0.02.

So the worry is not borne out on this window: once the intraday model can follow the day's level, it still finds
that single shocks lasted longer before the change than after. We expected the opposite ("weakened", with K between
0 and 0.5).

**But the finding is narrower than the paper's sentence.** Four things limit it.

1. **On the stated before window the fall disappears.** On 1 November 2023 to 29 February 2024 (the dates the paper
   states), the reference shows a fall from 383 to 72 minutes, but the fixed model shows 19.6 to 17.9 minutes for
   Nifty (bootstrap p 0.43, likelihood-ratio p 0.75) and a rise from 2.5 to 10.5 minutes for Bank Nifty. On that
   window the reference's fall is entirely due to the missing moving level.
2. **Bank Nifty is not confirmed.** On the inferred window the fixed model's Bank Nifty half-life falls from 62.7 to
   10.5 minutes. The likelihood-ratio test calls the fall significant (p 0.0003, Holm 0.001), but the bootstrap,
   which the plan trusts more, does not (interval -0.287 to +0.042, p 0.105, Holm 0.204): verdict "weakened". On
   P108's window, which shares 58 of its 64 days, the fixed model shows no fall at all (8.6 to 10.5 minutes).
3. **On P108's before window the Nifty result weakens.** The fixed model falls from 81 to 18 minutes, but the
   bootstrap interval reaches above zero (p 0.21, Holm 0.42): "weakened". Adding leverage (F4) on the inferred
   window also gives "weakened" (p 0.070, Holm 0.140).
4. **It is the before window that is unusual, not the after window.** Across 44 rolling 64-day Nifty windows from
   November 2022 to October 2026, the fixed model's half-life has a median of 25 minutes. The inferred before window
   has the largest of all 44 (141 minutes), while the three windows that end inside the after window have 19.5 to
   24.9. So under the fixed model, the fall is from an unusually persistent period at the end of 2022 to an ordinary
   level, not into a new, faster regime (exploratory, point estimates).

**The drift worry is real in general.** The drift measure (how much each day's volatility wandered from the
reference's daily level) is strongly rank-correlated with the reference's half-life across non-overlapping Nifty
windows (Spearman 0.88, p 0.0003), much less with the fixed model's (0.49, p 0.13). The reference finds no decay at
all (alpha + beta at its upper bound) in the windows around the June 2024 election result, where the fixed model's
half-life is 33 to 52 minutes. And in the inferred before window, each day's deviation from the reference's daily
level carried over to the next day far more than after (autocorrelation 0.63 against 0.21 for Nifty). So plain
mcsGARCH half-lives are inflated by drift; on the paper's window, though, the fall survives the fix.

**What it means.** The paper's finding of faster decay can be kept, stated more narrowly: for Nifty, on the window
that reproduces the paper's numbers, the fall survives a model with a moving intraday level, an overnight reset and
a better daily level. It is not confirmed for Bank Nifty or on the stated and P108 windows, its size is very
uncertain, and in a 2022 to 2026 view the before window looks unusual (Section 13).

## 2. What was registered (in short)

* **Models.** The reference: the replication's ARMA(1,1)-mcsGARCH(1,1) with Student t errors, daily level from an
  AR(1)-EGARCH(1,1), diurnal curve from the window itself, 360 minutes a day (09:15 to 15:14) with the overnight
  return in the first minute. The **fixed model F** (primary): X1 + X2 + X3. **F4** (secondary): F plus X4, the
  leverage term (falls raise volatility more than rises). **X1, X2, X3** alone: exploratory point estimates. X5
  (P109's time-of-day forecast factor) was left out because it does not enter the likelihood, so it changes no
  parameter and no impulse response.
* **Measures.** (a) alpha + beta, as in the paper. (b) **Primary**: the half-life of the impulse response to one
  standardised shock at 12:15, computed exactly from the model's own recursion (night = one step, the overnight
  fade applied, today's running mean restarting each day). Primary because the paper's claim is about how fast
  shocks die out, and in the fixed model part of a shock's effect runs through today's running mean and the overnight
  fade, which alpha + beta leaves out. For the reference model the two measures are the same thing.
* **Windows.** Inferred before window, 14 November 2022 to 10 February 2023, 64 days (primary); stated before
  window, 1 November 2023 to 29 February 2024, 82 days (secondary); P108's blind window, 22 November 2022 to 20
  February 2023, 64 days (robustness); after window, 1 March to 31 May 2024, 59 days. Nifty primary, Bank Nifty
  secondary.
* **Primary test** (P1, one test): the replication's day-block bootstrap (400 resamples of circular 5-day blocks,
  the replication's seeds, the diurnal curve and the model re-estimated on every resample, both models refitted on
  the same resampled days). Statistic D, its 95% percentile interval, and the one-sided bootstrap p for "persistence
  fell", (1 + draws with D at or above 0) / (1 + draws).
* **Decision rule** (Nifty, inferred window, model F): **survives** if the 95% interval of D is entirely below 0;
  **weakened** if D is below 0 but the interval reaches 0; **gone** if D is 0 or above and the interval reaches
  below 0; **reversed** if the interval is entirely above 0. With a size label from K: "keeps most of the fall"
  if K is at least 0.5, "keeps less than half" if K is between 0 and 0.5, "keeps none" if K is 0 or below.
* **Secondary families** (Holm within each): S1 Nifty inferred, F, the LR test and alpha + beta (3 tests); S2 Bank
  Nifty inferred, F (4); S3 the stated window, F (4); S4 F4 on the inferred window (2). **Robustness** R1: P108's
  window (2). **Exploratory** E1 (no adjustment): rolling 64-day windows from November 2022 and a drift measure
  stated in advance (4 tests). The likelihood-ratio (LR) test is secondary because it trusts the minute-by-minute
  likelihood.
* **What we expected** (stated in the plan, not a rule): "weakened", with a kept share between 0 and 0.5.

## 3. The primary result (REAL DATA, Nifty, inferred before window against the after window)

From `persistence.csv`, `persistence_change.csv`, `verdict.csv` and `tests.csv`. Bank Nifty (secondary) is shown
next to Nifty for comparison. Intervals are 95% bootstrap percentile intervals (400 draws each).

**Persistence and half-life in each window, both models:**

| Index, window | Reference: alpha + beta | Reference: half-life, minutes | Fixed model F: alpha + beta | F: half-life of a 12:15 shock, minutes |
|---|---|---|---|---|
| Nifty, before (inferred, 64 days) | 0.99807 [0.99331, 0.99901] | 358.6 [103.2, 700.8] | 0.99248 [0.96615, 0.99937] | **141.1** [26.2, 178.4] |
| Nifty, after (59 days) | 0.99048 [0.98124, 0.99504] | 72.4 [36.6, 139.3] | 0.95516 [0.92753, 0.97500] | **17.9** [10.4, 35.0] |
| Bank Nifty, before (inferred) | 0.99573 [0.98647, 0.99771] | 162.0 [50.9, 302.4] | 0.98556 [0.90432, 0.99913] | 62.7 [7.7, 178.4] |
| Bank Nifty, after | 0.98191 [0.93693, 0.99323] | 38.0 [10.6, 102.1] | 0.92838 [0.66404, 0.96918] | 10.5 [1.8, 27.3] |

For the reference, the half-life is ln(0.5) / ln(alpha + beta), the paper's half-life. For F it is the half-life
of the impulse response (Section 2), and F's alpha + beta leaves out the running-mean term and the overnight fade.
The upper ends 178.4 of F's before intervals are the **night cap**: in many draws (153 of 400 for Nifty, 83 of 400
for Bank Nifty; `posthoc_night_cap.csv`) more than half of a 12:15 shock's effect is still there at the close, and
only the overnight fade takes it below half, so the half-life stays just below 179 minutes (the largest among the
draws is 178.6). The cap shortens the before half-life in those draws, which makes the measured fall smaller, not
larger.

**The change from before to after:**

| Index | Model | half-life before to after (fall factor) | D = change in equivalent persistence [95%] | draws with a fall | bootstrap p, one-sided (Holm) | likelihood-ratio statistic, p (Holm) | registered label |
|---|---|---|---|---|---|---|---|
| Nifty | reference | 358.6 to 72.4 (4.95) | -0.0076 [-0.0163, -0.0006] | 99.0% | 0.012 | 16.23, 0.00006 (alpha + beta) | reference, not a new test |
| Nifty | **F** | **141.1 to 17.9 (7.88)** | **-0.0331 [-0.0579, -0.0026]** | **98.0%** | **0.022 (primary)** | 16.65, 0.00004 (0.0001) | **survives; keeps most of the fall** |
| Nifty | F4 | 65.3 to 20.9 (3.13) | -0.0221 [-0.0374, +0.0057] | 93.25% | 0.070 (0.140) | not registered | weakened; keeps most of the fall |
| Bank Nifty | reference | 162.0 to 38.0 (4.27) | -0.0138 [-0.0591, +0.0023] | 94.25% | 0.060 | 18.27, 0.00002 (alpha + beta) | reference, not a new test |
| Bank Nifty | F | 62.7 to 10.5 (5.98) | -0.0530 [-0.2867, +0.0422] | 89.75% | 0.105 (0.204) | 12.82, 0.0003 (0.001) | weakened; keeps most of the fall |
| Bank Nifty | F4 | 51.3 to 10.6 (4.85) | -0.0500 [-0.3073, +0.0276] | 91.5% | 0.087 (0.140) | not registered | weakened; keeps most of the fall |

The likelihood-ratio tests for F test equal half-life (two-sided, chi-square(1)); those for the reference are the
replication's tests of equal alpha + beta, recomputed by this study's code (Section 9).

**How much of the fall is kept** (paired draws: both models on the same resampled days):

| Index | log fall of the half-life under F [95%] | under the reference [95%] | kept share K [95%] | F's log fall minus the reference's [95%] |
|---|---|---|---|---|
| Nifty | 2.064 [0.107, 2.679] | 1.600 [0.125, 2.482] | **1.29** [-0.04, 6.91] | +0.46 [-1.42, +1.79] |
| Bank Nifty | 1.789 [-1.011, 3.508] | 1.451 [-0.208, 2.744] | 1.23 [-1.54, 10.83] | +0.34 [-1.73, +2.19] |

A log fall of 1.6 is a fivefold fall (358.6 / 72.4 = 4.95); 2.06 is about eightfold. K is the ratio of the two log
falls; the difference of the two is the steadier quantity, and its interval contains zero for both indices.

* **Verdict** (`verdict.csv`): **"survives", "keeps most of the fall (kept share at least one half)"**. D = -0.0331,
  interval -0.0579 to -0.0026, entirely below zero; K = 1.290 (paired interval -0.045 to 6.906).
* Under the registered reading (plan Section 5), "survives" with "keeps most of the fall" is the outcome that
  **supports** the paper's finding. The size label rests on the point estimate only; the K interval is too wide to
  say whether the fixed model keeps less or more of the fall than the reference shows.
* **alpha + beta under F** (S1): 0.9925 to 0.9552, change -0.0373 [-0.0649, -0.0015], 98.0% of draws fell, p 0.022.

**Post hoc check without the night cap** (`posthoc_night_cap.csv`, made by `experiments/p114_posthoc_night_cap.py`
after the results were seen; descriptive, no p-value). The share of a 12:15 shock's first effect still there at the
close (15:14) is computed for every fit and draw by the registered code; it does not depend on the overnight fade.

| Index, model | share at the close, before [95%] | after [95%] | change after minus before [95%] | draws with a fall |
|---|---|---|---|---|
| Nifty, reference | 0.709 [0.303, 0.839] | 0.182 [0.034, 0.412] | -0.527 [-0.710, -0.043] | 99.0% |
| Nifty, F | 0.444 [0.172, 0.906] | 0.147 [0.107, 0.211] | -0.297 [-0.737, -0.018] | 98.75% |
| Bank Nifty, F | 0.260 [0.071, 0.861] | 0.097 [0.036, 0.179] | -0.163 [-0.780, +0.077] | 86.5% |

It tells the same story as the registered measure: a clear fall for Nifty under F, not a clear one for Bank Nifty.

**The impulse responses** (`fig_irf.png`, `irf_curves.csv`, Nifty under F): after the change, a shock's effect
falls to 35% of its first value after 30 minutes and 20% after 60, then stays near 15% until the close (today's
running mean holds it), and the night removes most of the rest (2% left at the next open). Before the change it
decays slowly through the afternoon: 84% left after 30 minutes, 71% after 60, 44% at the close, 17% at the next
open.

## 4. What the fitted parameters say (REAL DATA)

From `persistence.csv` (the point estimates are also in `fits.csv`). F is the fixed model.

| | Nifty before (inferred) | Nifty after | Bank Nifty before (inferred) | Bank Nifty after |
|---|---|---|---|---|
| Reference: alpha (reaction to the last minute's squared return) | 0.0246 | 0.0498 | 0.0321 | 0.0538 |
| Reference: beta (carry of the last minute's variance) | 0.9735 | 0.9407 | 0.9636 | 0.9281 |
| F: alpha | 0.0282 | 0.0663 | 0.0383 | 0.0825 |
| F: beta | 0.9643 | 0.8889 | 0.9472 | 0.8459 |
| F: gam_x (weight on today's average squared return so far) | 0.0049 | 0.0395 | 0.0096 | 0.0589 |
| F: delta (share of the intraday part's distance from its average that survives the night) | 0.385 | 0.138 | 0.421 | 0.229 |
| F: alpha + beta | 0.9925 | 0.9552 | 0.9856 | 0.9284 |
| F: share of a 12:15 shock's first effect still there at the close (15:14) | 0.444 | 0.147 | 0.260 | 0.097 |
| F: share still there at the next open | 0.170 | 0.020 | 0.109 | 0.022 |
| F: half-life of a shock at 10:15 / 12:15 / 14:15, minutes | 298.0 / 141.1 / 58.3 | 28.5 / 17.9 / 16.7 | 118.5 / 62.7 / 55.9 | 13.8 / 10.5 / 10.0 |

In plain words:

* **Before the change, the fixed model still lets single shocks linger.** It carries most of each minute's
  variance into the next (Nifty beta 0.964) and gives today's running mean almost no weight (gam_x 0.005). So even
  a model that could follow the day's level with the running mean chose, on the before window, to explain the data
  with long-lived shocks.
* **After the change, the fixed model follows the day's level instead.** beta is lower (0.889) and the running mean
  gets eight times more weight (0.040). A single shock loses half of its effect in about 18 minutes, while the
  day's level is carried by the running mean.
* **The night.** Before, 38% of the intraday part's distance from its average survives the night (Nifty); after,
  14%. Bank Nifty: 42% and 23%.
* **The night cap** (plan Section 2). In the before window a Nifty shock at 10:15 is still above half of its first
  effect at the close, and only the overnight fade takes it below half (half-life 298.0 minutes, about the number
  of minutes from the shock to the close). A shock at 14:15 has a half-life of 58.3 minutes. The registered 12:15 shock lies between.
  After the change, the half-life hardly depends on the time of the shock (16.7 to 28.5 minutes).
* **alpha + beta alone** tells the same story for F: 0.9925 before and 0.9552 after (Nifty).
* The degrees of freedom of the Student t errors (nu) are about the same in both models (Nifty before 8.99 in the
  reference and 9.09 in F; after 7.01 and 6.91).

## 5. Secondary and robustness results (REAL DATA)

From `tests.csv`, `verdict.csv` and `persistence_change.csv`. Every row uses the fixed model's impulse-response
half-life unless it says alpha + beta. The bootstrap p is one-sided for "persistence fell"; the likelihood-ratio p
is two-sided. Holm is applied within each family.

| Test | Comparison | Estimate D [95%] | draws with a fall, or LR statistic | p | Holm p | label (`verdict.csv`) |
|---|---|---|---|---|---|---|
| S1 `S1_nifty_inferred_F_irf_lr` | Nifty, inferred, LR of equal half-life | -0.0331 | LR 16.65 | 0.00004 | 0.0001 | |
| S1 `S1_nifty_inferred_F_ab_boot` | Nifty, inferred, alpha + beta, bootstrap | -0.0373 [-0.0649, -0.0015] | 98.0% | 0.022 | 0.022 | |
| S1 `S1_nifty_inferred_F_ab_lr` | Nifty, inferred, LR of equal alpha + beta | -0.0373 | LR 15.84 | 0.00007 | 0.0001 | |
| S2 `S2_bank_inferred_F_irf_boot` | Bank Nifty, inferred, bootstrap | -0.0530 [-0.2867, +0.0422] | 89.75% | 0.105 | 0.204 | weakened |
| S2 `S2_bank_inferred_F_irf_lr` | Bank Nifty, inferred, LR | -0.0530 | LR 12.82 | 0.0003 | 0.001 | |
| S2 `S2_bank_inferred_F_ab_boot` | Bank Nifty, inferred, alpha + beta, bootstrap | -0.0572 [-0.2933, +0.0443] | 90.0% | 0.102 | 0.204 | |
| S2 `S2_bank_inferred_F_ab_lr` | Bank Nifty, inferred, LR of equal alpha + beta | -0.0572 | LR 12.53 | 0.0004 | 0.001 | |
| S3 `S3_nifty_stated_F_irf_boot` | Nifty, stated window, bootstrap | -0.0033 [-0.0345, +0.0218] | 57.0% | 0.431 | 1.000 | weakened; keeps less than half (K 0.06) |
| S3 `S3_nifty_stated_F_irf_lr` | Nifty, stated window, LR | -0.0033 | LR 0.10 | 0.749 | 1.000 | |
| S3 `S3_bank_stated_F_irf_boot` | Bank Nifty, stated window, bootstrap | +0.1765 [-0.1032, +0.2980] | 7.5% | 0.925 | 1.000 | gone; keeps none |
| S3 `S3_bank_stated_F_irf_lr` | Bank Nifty, stated window, LR | +0.1765 | LR 22.85 | 0.000002 | 0.000007 | (significant rise) |
| S4 `S4_nifty_inferred_F4_irf_boot` | Nifty, inferred, F4 | -0.0221 [-0.0374, +0.0057] | 93.25% | 0.070 | 0.140 | weakened |
| S4 `S4_bank_inferred_F4_irf_boot` | Bank Nifty, inferred, F4 | -0.0500 [-0.3073, +0.0276] | 91.5% | 0.087 | 0.140 | weakened |
| R1 `R1_nifty_p108_F_irf_boot` | Nifty, P108 window, bootstrap | -0.0294 [-0.0570, +0.1397] | 79.0% | 0.212 | 0.424 | weakened; keeps most (K 1.01) |
| R1 `R1_bank_p108_F_irf_boot` | Bank Nifty, P108 window, bootstrap | +0.0136 [-0.2127, +0.1686] | 56.75% | 0.434 | 0.434 | gone; keeps none |

**The other before windows, half-lives in minutes** (`persistence.csv`; the after window is the same as in Section 3:
Nifty 72.4 under the reference and 17.9 under F, Bank Nifty 38.0 and 10.5):

| Before window | Nifty: reference [95%] | Nifty: F [95%] | Bank Nifty: reference [95%] | Bank Nifty: F [95%] |
|---|---|---|---|---|
| inferred (primary) | 358.6 [103.2, 700.8] | 141.1 [26.2, 178.4] | 162.0 [50.9, 302.4] | 62.7 [7.7, 178.4] |
| stated (82 days) | 383.3 [72.7, at the bound] | 19.6 [12.6, 31.5] | 68.9 [22.4, 163.1] | 2.5 [1.6, 10.6] |
| P108 (64 days, 58 shared with inferred) | 324.9 [86.4, 718.4] | 81.1 [3.7, 178.4] | 138.0 [31.0, 313.6] | 8.6 [2.5, 178.3] |

"At the bound": the upper end of the reference's interval on the stated Nifty window is 64,777 minutes, that is,
alpha + beta at or near its upper limit in some draws.

In plain words:

* **S1: on the primary comparison, every other test and measure agrees** (all three Holm p at or below 0.022).
* **S2: Bank Nifty points the same way but the bootstrap cannot confirm it.** Only the likelihood-ratio tests are
  significant, and they trust the minute-by-minute likelihood, which ignores the dependence across days that the
  bootstrap respects.
* **S3: on the stated window, no fall.** Nifty's fixed-model half-life is almost the same before and after (19.6
  and 17.9 minutes). For Bank Nifty the fixed model's before fit is very different (beta 0.585, gam_x 0.175, a
  half-life of 2.5 minutes), and the likelihood-ratio test says persistence **rose** after the change (p 0.000002);
  the bootstrap gives no clear direction (interval -0.103 to +0.298). Under the reference this window shows a fall
  for Nifty (383 to 72 minutes, replication's LR p 0.0001) and a weak one for Bank Nifty (69 to 38, p 0.057).
* **S4: with leverage, the Nifty fall is smaller** (65.3 to 20.9 minutes) and the interval reaches zero.
* **R1: P108's window.** For Nifty, the fixed model's fall is the same size as the reference's in log terms (K
  1.01), but the bootstrap interval reaches far above zero: in some draws F's before half-life is very short (the
  2.5th percentile is 3.7 minutes). For Bank Nifty no fall at all. The reference on P108's window (descriptive, not
  a registered test) gives Nifty D = -0.0074 [-0.0164, -0.0001], 97.75% of draws fell, and Bank Nifty -0.0131
  [-0.0585, +0.0073], 92.5%.
* The post hoc share at the close (`posthoc_night_cap.csv`) agrees with all of these: no fall on the stated window
  (Nifty change +0.037 [-0.022, +0.105], Bank Nifty +0.066 [-0.004, +0.150]), and an unclear one on P108's window
  (Nifty -0.179 [-0.706, +0.099]).

## 6. One fix at a time (exploratory, REAL DATA, point estimates only)

From `persistence_change.csv` (inferred before window against the after window). Half-life of a 12:15 shock, in
minutes; the fall factor is before divided by after. No bootstrap and no test for X1, X2 and X3 (registered as
point estimates only).

| Model | Nifty before | Nifty after | fall factor | Bank Nifty before | Bank Nifty after | fall factor |
|---|---|---|---|---|---|---|
| reference | 358.6 | 72.4 | 4.95 | 162.0 | 38.0 | 4.27 |
| X1 HAR daily level only | 137.0 | 60.8 | 2.25 | 90.7 | 50.5 | 1.80 |
| X2 overnight fade only | 251.1 | 96.1 | 2.61 | 178.7 | 44.5 | 4.02 |
| X3 today's running mean only | 93.0 | 13.6 | 6.83 | 35.0 | 7.5 | 4.66 |
| **F = X1 + X2 + X3** | **141.1** | **17.9** | **7.88** | **62.7** | **10.5** | **5.98** |
| F4 = F + leverage | 65.3 | 20.9 | 3.13 | 51.3 | 10.6 | 4.85 |

In plain words:

* **The better daily level (X1) alone does what the worry predicted.** It takes most of the long memory out of the
  before window (Nifty 359 to 137 minutes) but little out of the after window (72 to 61), so the fall shrinks to a
  factor of about 2. The replication's daily EGARCH level was further off, day after day, in the before window.
* **Today's running mean (X3) alone shortens both windows and keeps the fall** (93 to 14 minutes for Nifty).
* **Together (F) the fall is as large as or larger than under the reference.** The fixes do not add up one by one:
  with the HAR level in place, the running mean gets less weight in the before window (gam_x 0.0049 in F against
  0.0086 with X3 alone), and the half-life goes back up to 141 minutes, while the after window stays short.
* **Adding leverage (F4)** lowers the Nifty before half-life to 65.3 minutes and the fall to a factor of about 3
  (Section 5, S4).
* Compared by likelihood (same daily level, same normalised returns), the overnight fade (X2) and the running mean
  (X3) each fit better than the reference on both windows: Nifty before, log-likelihood -30,859.7 (X2) and -30,865.5
  (X3) against -30,870.4; after, -28,327.3 and -28,309.5 against -28,343.4 (`fits.csv`).

## 7. Rolling windows and the drift measure (exploratory, REAL DATA)

From `rolling.csv`, `rolling_correlations.csv`, `persistence.csv` and `fig_rolling.png`. Windows of 64 trading
days, the first starting on 14 November 2022 (the inferred before window), then every 21 trading days: 44 Nifty
windows (the last ends on 16 October 2026) and 38 Bank Nifty windows (the last ends on 16 April 2026). Point
estimates only.

**The drift measure** (stated in the plan): for each day, the mean over its minutes of the squared normalised
return x^2 under the reference model, that is, how far the day's volatility stood above or below what the
reference's daily level and time-of-day curve expect. Drift is the standard deviation over the window's days of the
log of that mean; the plan also asked for its day-to-day (lag-1) autocorrelation.

| | Nifty before (inferred) | Nifty after | Bank Nifty before (inferred) | Bank Nifty after |
|---|---|---|---|---|
| drift, reference residuals | 0.501 | 0.486 | 0.483 | 0.407 |
| day-to-day autocorrelation, reference residuals | 0.633 | 0.211 | 0.592 | 0.111 |
| drift, F's residuals (HAR daily level) | 0.451 | 0.514 | 0.452 | 0.481 |
| day-to-day autocorrelation, F's residuals | 0.319 | -0.052 | 0.307 | -0.136 |

So in the before window the days' deviations from the reference's daily level were about as large as after, but
**much more persistent from one day to the next** (autocorrelation 0.63 against 0.21 for Nifty). That is exactly
the situation in which a model without a moving level needs a persistence near 1. The HAR daily level removes about
half of that day-to-day persistence (0.32).

**Half-lives across the rolling windows:**

| | Nifty: reference | Nifty: F | Bank Nifty: reference | Bank Nifty: F |
|---|---|---|---|---|
| smallest | 3.75 | 2.19 | 36.1 | 2.15 |
| median | 189.8 | 25.1 | 151.1 | 25.3 |
| largest | at the bound (6 windows) | 141.0 (window 0, the inferred before window) | at the bound (3 windows) | 107.9 (window starting 30 January 2025) |
| window 0 (14 November 2022 to 10 February 2023) | 359.0 | 141.0 | 164.9 | 62.2 |

"At the bound" means alpha + beta reached the model's upper limit 0.99999 (a half-life of about 69,000 minutes),
that is, the reference found no decay at all. This happens for Nifty in the three windows that start on 22 March,
26 April and 29 May 2024 (all contain the election result day, 4 June 2024) and in three windows starting 30
January to 3 April 2025; for Bank Nifty in the three 2024 election windows. F's half-lives in the same windows are
24 to 102 minutes.

* **Under F, the inferred before window is the most persistent of all 44 Nifty windows** (141.0 minutes, against a
  median of 25.1). The three Nifty windows that end inside the after window (starting 20 December 2023, 19 January
  and 21 February 2024) have 24.9, 19.5 and 21.0 minutes. For Bank Nifty window 0 (62.2 minutes) is the fourth
  largest of 38.
* So, under the fixed model, **the fall from before to after is a fall from an unusually persistent before window to
  an ordinary level**, not a move to an unusually fast after window. These are exploratory point estimates.
* Window 0 differs slightly from the Table 2 fit because the rolling windows use a one-index grid (plan Section
  12): reference 359.0 against 358.6 (Nifty) and 164.9 against 162.0 (Bank Nifty); F 141.0 against 141.1 and 62.2
  against 62.7.

**E1, drift against half-life** (`rolling_correlations.csv`, Spearman rank correlation; p-values only on the
non-overlapping windows, every fourth window):

| Index | Half-life of | non-overlapping windows | Spearman rho | p (two-sided) | all windows (descriptive) |
|---|---|---|---|---|---|
| Nifty | reference | 11 | **0.882** | **0.0003** | 0.835 (44 windows) |
| Nifty | F | 11 | 0.491 | 0.125 | 0.577 (44) |
| Bank Nifty | reference | 10 | 0.491 | 0.150 | 0.735 (38) |
| Bank Nifty | F | 10 | 0.067 | 0.855 | 0.326 (38) |

The plan said that if the artefact story is right, the correlation with the reference's half-life should be
clearly positive and the one with the fixed model's weaker. That is what happens for Nifty: windows in which the
day's level wandered more are the windows in which the reference finds long-lived shocks, and the link is much
weaker under F. For Bank Nifty both correlations are weak on the 10 non-overlapping windows, although over all 38
windows the pattern is the same (0.735 against 0.326). Exploratory, no adjustment.

## 8. What we expected, and what happened (REAL DATA)

| Registered expectation (plan Section 5) | Outcome |
|---|---|
| both windows have much shorter half-lives under F than under the reference (tens of minutes rather than hundreds) | **mostly as expected**: after, 17.9 and 10.5 minutes (reference 72.4 and 38.0); Bank Nifty before 62.7 (162.0); but Nifty before is still 141.1 under F, less than half of the reference's 358.6 |
| the fall, if any, is smaller than the reference's in log terms | **not as expected**: larger in the point estimate (2.06 against 1.60 for Nifty, 1.79 against 1.45 for Bank Nifty); the difference is not significant (intervals -1.42 to +1.79 and -1.73 to +2.19) |
| verdict "weakened", with a kept share between 0 and 0.5 | **not as expected**: "survives", kept share 1.29 (interval -0.04 to 6.91) |
| E1: the drift is clearly correlated with the reference's half-life and less with F's | **as expected for Nifty** (0.88, p 0.0003, against 0.49, p 0.13); for Bank Nifty both are weak on non-overlapping windows (0.49 and 0.07) |

## 9. Checks (REAL DATA unless marked SYNTHETIC)

* **Inputs.** The three raw files the study reads (`nifty_1min.csv`, `banknifty_1min.csv`,
  `minute_bars_source.json`) match `data/MANIFEST.csv` (`run_info.json`, `inputs`; `run_stdout_inputs.txt`).
* **Table 2 reproduces exactly** (`reproduction_table2.csv`): the reference fitted by this study's driver gives 36
  of 36 committed values identical to the last bit (omega, alpha, beta, nu and the AR and MA terms, on the three
  windows and two indices; `run_stdout_fit.txt`).
* **The replication's bootstrap reproduces exactly** (`reproduction_bootstrap_lr.csv`): all 20 committed bootstrap
  numbers of the reference (12 percentiles of alpha + beta and 8 ends of the intervals of its change) are identical
  to the last bit. So this study's bootstrap is the replication's bootstrap, and the fixed model was refitted on
  exactly the replication's resampled days.
* **The replication's likelihood-ratio statistics reproduce to four decimals** with this study's profile code and
  every fix switched off: 16.229945, 18.272162, 14.381910 and 3.629035 against the committed 16.229945, 18.272192,
  14.381911 and 3.629035 (largest difference 0.00003, inferred Bank Nifty). They are not identical to the last bit
  (the file marks them `False`) because this study's profile uses a different optimiser path (a Nelder-Mead
  polish); the plan asked for agreement to four decimals.
* **The ARMA residuals of all six models are identical** on every window (checked by `fit`, which stops otherwise).
* **Convergence.** All 48 point fits, all 164 rolling fits (44 + 38 windows, two models each) and every bootstrap
  refit (share converged 1.0 for every model, window and index) converged. No bootstrap draw lacked a measure
  (`boot_draws_not_computed` 0 everywhere).
* **P109's forecasts are unchanged** by this study's new shared code (phase 1, `plan_checks_stdout.txt`): two Nifty
  refit blocks recomputed, 18,900 of 18,900 and 13,500 of 13,500 cells identical.
* **The code of the run is the plan's code**: `analysis_commit` = `plan_commit` = `86e3866...` in `run_info.json`.
* **Tests.** `tests/test_p114_persistence_fixed_model.py` (SYNTHETIC only, 18 tests, committed with the plan)
  passes. The full suite (`python -m pytest -q tests/`), run from 10:48:31 to 10:54:31 UTC before the phase-2
  commit with the raw data linked in `data/raw` (so the tests that need raw data ran too): 658 tests, all passed,
  none skipped or failed (exit code 0). The project's `pyproject.toml` already adds `-q`, so pytest printed no
  summary line; the count is that of the progress dots.

## 10. Run time

On 3 November 2026, one worker process (`run_times.txt`, `run_stdout_*.txt`, `run_info.json`, `lr.csv`):

* inputs 10:11:42 to 10:11:44 UTC; fit 26 seconds (to 10:12:10; 24.8 seconds inside the stage); likelihood-ratio
  tests 48 seconds (to 10:12:58); rolling windows 68 seconds (to 10:14:06);
* bootstrap 33.3 minutes (10:14:06 to 10:47:24): 342, 298, 338 and 303 seconds for the four keys with F4
  (inferred Nifty, after Nifty, inferred Bank Nifty, after Bank Nifty), then 192, 204, 152 and 168 seconds for the
  stated and P108 keys;
* analysis 2.2 seconds (10:47:39); post hoc script a few seconds (10:48).

About 36 minutes in all. The full test suite took 6.0 minutes.

## 11. Deviations from the plan

1. **None in the design.** Every registered model, window, measure, test, family and output was run as written,
   with no change to the registered code after the plan commit (`analysis_commit` in `run_info.json` is the plan
   commit, and the phase-2 commit changes no file of the plan commit; it only adds the post hoc script of item 4). No
   run was restarted, no bootstrap key was recomputed, and the bootstrap finished at 10:47 UTC, long before the
   16:30 UTC cut-off, so nothing was left out. No second worker was needed.
2. **Commands.** The plan's commands set `PYTHONPATH=src`; the run used the absolute path of the same `src` folder
   and also set `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. The five stages ran in the registered order
   (inputs, fit, lr, rolling, boot) from one wrapper script started with `setsid nohup`, so that it was not tied to
   the agent's session; `analyze` was then run with `--plan-commit 86e386635284b24d876c32a7aecc63a1ecd23886`. Same
   code.
3. **Git housekeeping (no effect on any number).** While phase 1 was being prepared, another agent's commits (P113)
   moved the shared local branch name that this worktree had checked out, which left the working copy's index out
   of step with it. The AI agent synced the index with `git read-tree -m -u` (a safe two-way merge that changed no
   file of this study), then did all further work on a private local branch, `p114-persistence`, that tracks
   `origin/claude/project-thread-rbrfe4`. Both commits were pushed to `claude/project-thread-rbrfe4` as instructed.
4. **One post hoc addition** (decided after the results were seen, because 153 of the 400 Nifty before-window draws
   of F had their half-life at the night cap): `experiments/p114_posthoc_night_cap.py` writes
   `posthoc_night_cap.csv`, which counts those draws and summarises a measure without the cap, the share of a
   shock's first effect left at the close (already computed for every fit and draw by the registered code). It is
   descriptive: no p-value, nothing added to `tests.csv`, and it changes no registered number or verdict.
5. **Reading notes, not deviations.** The kept share K is a ratio; it is read only through its point estimate and
   paired interval, and where the reference's own fall is small (the stated Bank Nifty window) it is not read. The
   likelihood-ratio tests are two-sided, so the significant S3 Bank Nifty row is a rise in persistence, not a fall.
6. **Tests.** The 18 synthetic tests were committed with the plan (phase 1); none was added in phase 2.

## 12. Caveats

1. **Small windows.** 64 and 59 days. Long half-lives are poorly determined: the fixed model's Nifty before
   half-life has a bootstrap interval of 26.2 to 178.4 minutes (the upper end is the night cap, caveat 6), and the
   kept share's interval runs from -0.04 to 6.91. The primary test is about the direction of the change, which is
   much better determined than its size.
2. **In sample, like the paper.** Every parameter comes from the window it describes. No forecasts are evaluated.
3. **The fixes were chosen for forecasting** (P109, on 2023 to 2026 data), not for this question. Other ways to give
   the intraday model a moving level (a two-component intraday GARCH, a realized GARCH) were not tried, and could
   give other half-lives.
4. **Bank Nifty's fixed model is unstable across nearly the same days.** The inferred and P108 windows share 58 of
   64 days, yet F's Bank Nifty before half-life is 62.7 minutes on one and 8.6 on the other (the reference's: 162.0
   and 138.0). The P108 Bank Nifty fit puts much more weight on the running mean (gam_x 0.064 against 0.010).
   Bank Nifty results under F should be read as fragile.
5. **The half-life of F mixes a fast and a slow part.** A shock's effect runs through the last minute's variance
   (fast, alpha and beta) and through today's running mean (slow, gam_x). When gam_x is large, the half-life is set
   by the fast part, and the slow part shows only in the share left at the close. This is why the plan reports both.
6. **The night cap.** With a small delta, a mid-day shock cannot have a half-life much beyond the close, because the
   overnight fade removes most of what is left (Section 4). For the registered 12:15 shock it does not bind at the
   point estimates (Nifty F before: 141.1 minutes, with the close 179 minutes after the shock), but it does in 153
   of the 400 Nifty before-window draws (83 for Bank Nifty; `posthoc_night_cap.csv`), which is why F's before
   intervals end at 178.4 minutes. In those draws the cap shortens the before half-life and so makes the measured
   fall smaller; the share left at the close, which has no cap, gives the same answer (Section 3). It also binds for
   a Nifty shock at 10:15 in the before window (Section 4).
7. **The likelihoods of the reference and F are not comparable**, because F's daily level (HAR) gives different
   normalised returns. Only models with the same daily level (reference, X2, X3) can be compared by likelihood.
8. **The bootstrap keeps the full-sample ARMA residuals and daily level fixed** and joins resampled days at block
   boundaries, as the replication does. The LR test's profile uses one start and a bounded search.
9. **Rolling windows** use one-index grids (plan Section 12), and in 6 Nifty and 3 Bank Nifty windows the reference's
   alpha + beta reaches the upper bound 0.99999, so its half-life (about 69,000 minutes) means "no decay found", not a
   measured value. Rolling results are point estimates without intervals, and overlapping windows are not
   independent.
10. **Which before window.** The paper's numbers match the inferred window; its stated dates give the stated window.
   Under the reference both show a fall; under F only the inferred window does (Section 5).

## 13. What this means for the paper's claim

* **The registered answer supports the paper on its own window.** On the inferred before window, which reproduces
  the paper's numbers, Nifty's faster decay of intraday shocks after the change survives a model that has a moving
  intraday level (X3), an overnight reset (X2) and a better daily level (X1). The specific worry of this study,
  that the fall is only a symptom of mcsGARCH's missing moving level, is not supported there.
* **The sentence should be narrower than "shocks die out five times faster".** What can be said with this
  evidence: "For Nifty, on the before window that reproduces the paper's numbers, the half-life of an intraday
  volatility shock falls from about 141 to 18 minutes in a model with P109's fixes (359 to 72 in the paper's
  model); 98% of bootstrap draws show a fall." The size is very uncertain (kept share interval -0.04 to 6.91).
* **It does not generalise well.** Bank Nifty is not confirmed by the bootstrap; on P108's window (58 of 64 days in
  common) Nifty weakens and Bank Nifty shows no fall; on the stated window there is no fall under the fixed model at
  all, and for Bank Nifty a rise. The stated-window fall that the paper's model shows is an artefact of the missing
  moving level.
* **The before window is the outlier.** In 44 rolling Nifty windows from November 2022 to October 2026, the fixed
  model's half-life is largest in the inferred before window and ordinary in the windows around the after window.
  Nothing in this study can tell a lasting effect of the expiry change apart from a temporarily persistent period
  at the end of 2022; a claim that the change *caused* faster decay needs that comparison (for example a
  difference-in-differences against windows with no change, or more before windows).
* **For the project's other work.** Plain mcsGARCH half-lives move with the day-level drift (E1, Nifty) and can hit
  the persistence bound around large events (June 2024), where the fixed model gives 33 to 52 minutes. Any later
  persistence comparison in this project should be run with a moving intraday level as well as with the paper's
  model.
* These statements are for the lead to weigh; this study does not edit the project's summary documents.

## 14. Files

All in `results/p114_persistence_fixed_model/` (REAL DATA):

* `tests.csv`: every p-value, with the registered columns.
* `verdict.csv`: the primary verdict and the same rule applied, as a description, to every other comparison of F
  and F4, with kept shares and their intervals.
* `persistence.csv`: every model, window and index: parameters, alpha + beta, both half-lives, equivalent
  persistence, shares left at the close and the next open, half-lives for shocks at 10:15 and 14:15, the drift
  measures, and the bootstrap intervals.
* `persistence_change.csv`: every before-after comparison: changes, bootstrap intervals and p, fall factors, log
  falls, kept shares with paired intervals, LR results.
* `fits.csv`, `lr.csv`: the point fits and the likelihood-ratio tests (with the reference's LR checks).
* `bootstrap_draws.csv`: all bootstrap draws (every model, window, index and draw), from which every interval can be
  recomputed.
* `rolling.csv`, `rolling_correlations.csv`: the rolling windows and the E1 correlations.
* `irf_curves.csv`, `fig_irf.png`: the impulse responses over three trading days after a 12:15 shock;
  `fig_rolling.png`: half-lives and drift across the rolling windows.
* `reproduction_table2.csv`, `reproduction_bootstrap_lr.csv`: the reproduction checks of Section 9.
* `run_info.json`: plan commit, analysis commit, settings, input hashes, bootstrap timings and versions;
  `run_times.txt`: the clock time at the end of each stage.
* `posthoc_night_cap.csv` (**post hoc**, descriptive): draws whose response is still above half at the close, the
  largest half-life among the draws, and the share left at the close with its intervals, for every comparison of
  the reference, F and F4.
* `plan_checks_stdout.txt` (phase 1), `run_stdout_inputs.txt`, `run_stdout_fit.txt`, `run_stdout_lr.txt`,
  `run_stdout_rolling.txt`, `run_stdout_boot.txt`, `analyze_stdout.txt`, `posthoc_night_cap_stdout.txt`: logs.

The work files (one bootstrap file per window and index, `fits.pkl`) are in `data/processed/p114/` (git-ignored).

## 15. How to reproduce

From the repository root, with the raw data listed in `data/MANIFEST.csv` in `data/raw`:

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py inputs
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py fit
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py lr
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py rolling
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py boot --workers 1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py analyze \
    --plan-commit 86e386635284b24d876c32a7aecc63a1ecd23886
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_posthoc_night_cap.py     # post hoc, after analyze
```

About 36 minutes on one core (Section 10). The seeds are fixed per window and index, so more workers give the
same numbers. The phase-1 checks: `PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_plan_checks.py`. The
synthetic tests: `PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_p114_persistence_fixed_model.py`.
A synthetic end-to-end run of all stages: add `--synthetic` to each command (labelled SYNTHETIC).
