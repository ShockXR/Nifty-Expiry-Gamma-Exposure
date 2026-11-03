# P118: the hybrid built on mcsGARCH as the paper specifies it

Study P118, run by an AI agent (Claude) working for Shreyan Thanki on 3 November 2026.
**All numbers below are REAL DATA** (Nifty, Bank Nifty and Fin Nifty, NSE one-minute index data as in the
headline runs) unless marked SYNTHETIC. Every number comes from a file in `results/p118_hybrid_on_paper_mcsgarch/`
(file named next to each table) or, where marked, from an earlier committed results file.

* Registered plan: `docs/prereg/P118_HYBRID_ON_PAPER_MCSGARCH.md`, commit `49d19ce2779451e13cbf868d325a651c4da6a1f4`
  (committed at 12:16 UTC and pushed at about 12:17 UTC, before any forecast of the new model was made on real data
  and before any loss of this study was computed).
* Code (all in the plan commit, unchanged since): `experiments/p118_hybrid_on_paper_mcsgarch.py`; an optional code
  path in `src/niftygex/forecast.py` (`WalkForwardConfig.alt_base` and `WalkForwardConfig.mcs_mean`, whose defaults
  run the committed code). Tests (SYNTHETIC only): `tests/test_p118_hybrid_on_paper_mcsgarch.py`.
* Every p-value of the study: `results/p118_hybrid_on_paper_mcsgarch/tests.csv` (75 rows: 4 primary, 25 secondary,
  8 robustness, 16 placebo, 22 exploratory).

**Names.** `mcs_ours` is our implementation of mcsGARCH (the project's benchmark until P116: a separate time-of-day
curve for expiry days, a 5-minute smoothing of that curve, a zero expected return). `mcs_cpp` is mcsGARCH as the
anchor paper (Chhimwal, Pandey and Pandey 2025) specifies it (one time-of-day curve, no smoothing, an ARMA(1,1)
expected return, Student t errors), as refitted in P116. The **committed hybrid** (`hybrid`) is the project's model:
mcs_ours x a LightGBM correction learnt from 41 inputs. **`hybrid_cpp`** is the identical recipe with `mcs_cpp` in
place of `mcs_ours`, both as the starting point the correction multiplies (the "base") and as the input `log_mcs`.

**Units.** A **gain** of model m over a benchmark b is 100 x (L_b - L_m) / L_b, in % of the benchmark's mean daily
QLIKE (lower QLIKE is better; positive gain means m is better). "Points" are % of `mcs_cpp`'s mean daily QLIKE.

* **G**: `hybrid_cpp`'s gain over `mcs_cpp`, %.
* **E**: `hybrid_cpp`'s gain over the committed hybrid, points. **Positive means the hybrid built on the paper's
  model is better.**
* **B**: the difference of the two bases, `mcs_cpp`'s gain over `mcs_ours`, points (1.45 and 1.09 on Nifty).
* **A = 1 - E / B**: the share of the difference between the bases that the correction absorbs (1: the two hybrids
  are equally good; 0: the hybrids differ by as much as their bases).

Unless a row says "seed set 7", every hybrid number on Nifty, Bank Nifty and Fin Nifty at 15 and 30 minutes is the
registered **primary sample: each hybrid's daily QLIKE averaged over the five seed sets** (7, 101, 102, 103, 104).
The last hour has seed set 7 only.

## 1. The answer in plain words (REAL DATA)

Our hybrid forecasts how much the Nifty will move in the next 15 or 30 minutes. It starts from mcsGARCH's forecast
and multiplies it by a correction that LightGBM learns. Until now the starting point was our own version of
mcsGARCH. P116 found that the paper's own specification forecasts Nifty a little better (by 1.45% and 1.09% of its
loss at 15 and 30 minutes). This study rebuilt the hybrid, with nothing else changed, on the paper's specification.

**Question 1. How much better is the paper's model plus the learnt correction than the paper's model alone?**

* **5.06% better at 15 minutes and 7.09% better at 30 minutes** (lower QLIKE than `mcs_cpp`; one-sided
  Diebold-Mariano, Holm-adjusted p below 0.001 at both; every one of the five seed sets beats `mcs_cpp`, by 4.93% to
  5.22% at 15 minutes and 7.00% to 7.24% at 30 minutes).
* In the last hour (seed set 7) it is 5.82% better (Holm p below 0.001).
* This is the like-for-like statement for the professor: the paper's model as the paper specifies it, plus a learnt
  correction, against the same model without the correction.

**Question 2. Does the correction absorb the choice of base, so that both hybrids are equally good?**
**The data cannot tell. The registered verdict is "inconclusive" at both horizons.**

* The hybrid built on the paper's model is a little better than the committed hybrid: by **0.65 points at 15
  minutes** (95% interval 0.05 to 1.24) and **0.25 points at 30 minutes** (-0.30 to 0.81). All five seed sets point
  the same way at both horizons.
* Equivalence within the registered 0.5-point margin is **not shown** (TOST, Holm p 0.679 and 0.355). A
  difference is **not shown** either after the registered Holm adjustment (two-sided DM, raw p 0.040 and 0.344,
  Holm p 0.079 and 0.344). At 15 minutes the estimate (0.65) is just above the margin and the raw test is just below
  5%, so it leans towards "the paper's base gives a slightly better hybrid", but the registered rule does not let us
  say so.
* **The correction absorbs about half to three quarters of the difference between the bases**: A = 0.56 at 15
  minutes (95% interval 0.17 to 0.95) and 0.77 at 30 minutes (0.20 to 1.62). So the base's advantage of 1.45 and 1.09
  points shrinks to 0.65 and 0.25 points once the correction is on top. Neither "nothing absorbed" (A = 0) nor, at 15
  minutes, "all absorbed" (A = 1) fits the interval.

**Where the base still matters: expiry days.**

* On the 169 expiry days `hybrid_cpp` is **2.05 points better at 15 minutes** (S2, Holm p 0.028, all five seed sets
  agree: a registered difference) and 1.37 points at 30 minutes (Holm p 0.228, not shown). On the 623 other days the
  two hybrids are almost the same: +0.17 and -0.12 points (exploratory, p 0.556 and 0.602).
* This is what the plan expected: the noise of our separate expiry-day curve (estimated from about one day in five)
  passes through the correction, while the flaw of our 5-minute smoothing, the same on every day, is mostly learnt
  away. (This is an interpretation that fits the split by day type; it is not a separate test.)

**Each hybrid against its own base.** The committed hybrid gains 5.79% and 7.84% over `mcs_ours`; `hybrid_cpp`
gains 5.06% and 7.09% over `mcs_cpp`. The gain over its own base is 0.72 and 0.75 points smaller (95% intervals
-1.23 to -0.11 and -1.39 to -0.08): a better base leaves less for the correction to do, but the final forecast is
not worse.

**Bank Nifty and Fin Nifty** (secondary, nothing retuned; the decision rule applied descriptively):

* On **Bank Nifty**, where our implementation is the better base (P116), `hybrid_cpp` beats `mcs_cpp` by 5.01% and
  6.16% (Holm p below 0.001). At 15 minutes the two hybrids are **shown to be equivalent** within 0.5 point (E
  +0.11, 95% interval -0.22 to 0.45; TOST Holm p 0.028, all five seed sets inside the margin): there the correction
  absorbs the base. At 30 minutes E is -0.26 (-0.66 to 0.14) and neither equivalence nor a difference is shown. So
  Bank Nifty's outcome is "mixed".
* On **Fin Nifty** `hybrid_cpp` beats `mcs_cpp` by 4.47% and 5.69% (Holm p below 0.001) and is 0.50 and 0.55 points
  better than the committed hybrid (95% intervals 0.13 to 0.88 and -0.01 to 1.10; all five seed sets agree). After
  Holm neither equivalence nor a difference is shown (difference Holm p 0.057 and 0.128): "inconclusive".
* **On every index and at both horizons, the hybrid built on the paper's model beats the paper's model.**

**What it means.** The cleanest figure for the professor is now **5.06% (15 minutes) and 7.09% (30 minutes)**: the
paper's model plus the learnt correction against the paper's model, averaged over five seed sets. The committed
hybrid's gain over `mcs_cpp` (4.42% and 6.83% over five seed sets; 4.43% and 6.64% for seed set 7, as in P116) is
slightly smaller, and the data cannot say whether the 0.65 and 0.25 point difference between the two hybrids is
real. Section 10 gives the wording.

## 2. What was registered (in short)

* **Models.** `mcs_ours` and `mcs_cpp` (both refitted inside every job and checked identical to the committed and
  P116 forecasts); the committed hybrid (seed set 7 from the committed forecasts, seed sets 101 to 104 from P106's
  and P110's committed daily losses); **`hybrid_cpp`** with seed sets 7, 101, 102, 103 and 104; `hybrid_cpp_all`
  (exploratory, seed set 7: every mcsGARCH of the walk-forward as the paper specifies it, so that `log_s_window`,
  `log_mcs_iv` and `log_implied_window` also come from the paper's specification); P109's combination from its
  committed daily losses.
* **Design.** The committed walk-forward: the first 150 days for training only, then a refit on all earlier days
  before every block of 21 trading days. Nifty 792 test days (10 August 2023 to 2 November 2026), Bank Nifty 660,
  Fin Nifty 604; anchors every 15 minutes; the last hour at 14:30. The same 41 inputs, untuned LightGBM settings and
  five inner LightGBM seeds per seed set as the committed hybrid.
* **Primary tests** (4 rows, Holm over all four), Nifty, 15 and 30 minutes, five-seed average: H1, `hybrid_cpp`
  beats `mcs_cpp` (one-sided DM); H2, `hybrid_cpp` and the committed hybrid are equivalent within 0.5 point (TOST).
* **Seed rule** (P110): a five-seed row passes only if its Holm p is below 0.05 **and** every seed set alone points
  the same way; otherwise "seed-fragile".
* **Decision rule** at each horizon, from the H2 row and the S0 difference row (two-sided DM, Holm over 2):
  "correction absorbs the base" (H2 passes); "base matters, cpp better" or "base matters, ours better" (H2 does not
  pass, S0 passes, by the sign of E); "inconclusive" (neither passes). "Mixed" if the horizons differ.
* **Secondary families** (Holm within each): S0 the difference (2), S1 last hour (3), S2 expiry days (6), S3 P109's
  combination (2), S4 Bank Nifty (6), S5 Fin Nifty (6). **Robustness**: R1 seed set 7 alone (6), R2 delta-method
  TOST (2). **Placebo**: PL1 `hybrid_cpp`'s seed sets against each other (8), PL0 the committed hybrid's (8, known
  from planning). **Exploratory** (no adjustment): E1 `hybrid_cpp_all` (9), E2 non-expiry days (6), E3 the 09:15
  anchor and the rest (4), E4 against `mcs_ours` (3).

## 3. The main result (REAL DATA, Nifty, 792 test days)

From `losses.csv`, `tests.csv`, `verdict.csv` and `absorption.csv`. Mean daily QLIKE (lower is better) and gains in
% of each benchmark's QLIKE; hybrids at 15 and 30 minutes are five-seed averages, in the last hour seed set 7.

| Model | 15 min: QLIKE | gain vs `mcs_cpp` | gain vs `mcs_ours` | 30 min: QLIKE | gain vs `mcs_cpp` | gain vs `mcs_ours` | last hour: QLIKE | gain vs `mcs_cpp` | gain vs `mcs_ours` |
|---|---|---|---|---|---|---|---|---|---|
| `mcs_ours` (our implementation) | 0.2845 | -1.45 | 0 | 0.2176 | -1.09 | 0 | 0.2395 | -0.22 | 0 |
| `mcs_cpp` (the paper's specification) | 0.2804 | 0 | +1.43 | 0.2152 | 0 | +1.08 | 0.2390 | 0 | +0.22 |
| committed hybrid (built on `mcs_ours`) | 0.2680 | +4.42 | +5.79 | 0.2005 | +6.83 | +7.84 | 0.2243 | +6.16 | +6.36 |
| **`hybrid_cpp`** (built on `mcs_cpp`) | **0.2662** | **+5.06** | +6.42 | **0.2000** | **+7.09** | +8.09 | 0.2251 | +5.82 | +6.02 |
| `hybrid_cpp_all` (exploratory, seed set 7) | 0.2656 | +5.30 | +6.66 | 0.1998 | +7.14 | +8.14 | 0.2250 | +5.85 | +6.05 |
| P109's combination (committed) | 0.2651 | +5.48 | +6.83 | 0.1996 | +7.25 | +8.25 | 0.2376 | +0.58 | +0.80 |

**The registered tests** (`tests.csv`; estimate, DM or TOST statistic, raw p, Holm p within the family; the label
applies the seed rule):

| Test | 15 min | 30 min |
|---|---|---|
| H1: `hybrid_cpp` beats `mcs_cpp` (one-sided; estimate G, %) | +5.06, stat 6.44, p below 0.001 (Holm below 0.001): **supported** | +7.09, stat 9.24, p below 0.001 (Holm below 0.001): **supported** |
| H2: `hybrid_cpp` and the committed hybrid equivalent within 0.5 point (TOST; estimate E, points) | +0.65, p 0.679 (Holm 0.679): not shown | +0.25, p 0.177 (Holm 0.355): not shown |
| S0: they differ (two-sided; estimate E) | +0.65, stat 2.06, p 0.040 (Holm 0.079): no significant difference | +0.25, stat 0.95, p 0.344 (Holm 0.344): no significant difference |

**E, B and A** (`absorption.csv`, `verdict.csv`; 95% stationary-bootstrap intervals, 10,000 draws):

| | 15 min | 30 min |
|---|---|---|
| G: `hybrid_cpp`'s gain over `mcs_cpp` | 5.06 | 7.09 |
| the committed hybrid's gain over `mcs_cpp` | 4.42 | 6.83 |
| **E** = G minus that [95% interval] | **+0.65 [0.05, 1.24]** | **+0.25 [-0.30, 0.81]** |
| B: the bases' difference [95%] | 1.45 [0.62, 2.30] | 1.09 [0.25, 1.92] |
| **A** = 1 - E/B, share absorbed [95%] | **0.56 [0.17, 0.95]** | **0.77 [0.20, 1.62]** |
| each hybrid against its own base: G minus the committed hybrid's gain over `mcs_ours` [95%] | -0.72 [-1.23, -0.11] | -0.75 [-1.39, -0.08] |
| outcome | inconclusive | inconclusive |

* **Verdict: "inconclusive"** at both horizons (neither equivalence nor a difference is shown). H1: "`hybrid_cpp`
  beats `mcs_cpp` at every primary horizon".
* **How wide a margin would equivalence need?** The smallest margin the TOST would pass is 1.16 points at 15 minutes
  and 0.69 points at 30 minutes (`equivalence_bounds.csv`; the 90% interval of E is 0.13 to 1.16 and -0.19 to 0.69).
  So the two hybrids are shown to be within about 1.2 points (15 minutes) and 0.7 points (30 minutes) of each other,
  not within 0.5.
* At 15 minutes the bootstrap interval of E (0.05 to 1.24) just excludes zero and the raw DM p is 0.040, while the
  registered Holm-adjusted p is 0.079. The registered rule uses the Holm p, so no difference is claimed. A reader
  should read this as weak evidence that the paper's base gives a slightly better hybrid at 15 minutes.

**Seed sets one by one** (`seed_repeats.csv`; the committed hybrid's seed sets 101 to 104 from P106's committed
daily losses):

| Seed set | 15 min: `hybrid_cpp` G | committed hybrid's gain vs `mcs_cpp` | E | 30 min: `hybrid_cpp` G | committed hybrid's gain vs `mcs_cpp` | E |
|---|---|---|---|---|---|---|
| 7 | 4.95 | 4.43 | +0.52 | 7.00 | 6.64 | +0.36 |
| 101 | 4.93 | 4.27 | +0.66 | 7.12 | 6.81 | +0.30 |
| 102 | 5.22 | 4.46 | +0.76 | 7.02 | 7.01 | +0.01 |
| 103 | 5.04 | 4.42 | +0.62 | 7.24 | 6.94 | +0.29 |
| 104 | 5.18 | 4.52 | +0.66 | 7.06 | 6.76 | +0.30 |
| five-seed average (primary) | 5.06 | 4.42 | +0.65 | 7.09 | 6.83 | +0.25 |

Every seed set's `hybrid_cpp` beats `mcs_cpp`, and every seed set's E is positive (0.52 to 0.76 at 15 minutes, 0.01
to 0.36 at 30 minutes): the seed rule is met for H1 and for the sign of S0; the S0 and H2 rows fail on their Holm p,
not on the seeds.

## 4. Robustness and placebo (REAL DATA, Nifty)

**R1, seed set 7 alone** (the literal headline seed set against itself; Holm over 6) and **R2, the delta-method
TOST** (Holm over 2) (`tests.csv`, `equivalence_bounds.csv`):

| Test | 15 min | 30 min |
|---|---|---|
| R1 H1 (G) | +4.95, Holm p below 0.001: supported | +7.00, Holm below 0.001: supported |
| R1 TOST (E) | +0.52, Holm p 0.746: not shown (smallest passing margin 0.95) | +0.36, Holm 0.746: not shown (0.87) |
| R1 difference (E) | +0.52, raw p 0.046, Holm 0.185: no significant difference | +0.36, raw 0.249, Holm 0.746: no significant difference |
| R2 delta-method TOST (E) | +0.65, Holm p 0.689: not shown (1.13) | +0.25, Holm 0.357: not shown (0.69) |

The verdict does not depend on the seed averaging or on the variance method: with seed set 7 alone or with the
delta method, neither equivalence nor a difference is shown.

**PL1, placebo** (`hybrid_cpp` with seed set s against `hybrid_cpp` with seed set 7; only the LightGBM seeds differ;
TOST at 0.5 point, Holm over 8):

| Seed set | 15 min: estimate, Holm p | 30 min: estimate, Holm p |
|---|---|---|
| 101 | -0.02, below 0.001: equivalent | +0.11, 0.003: equivalent |
| 102 | +0.27, 0.088: not shown | +0.01, below 0.001: equivalent |
| 103 | +0.09, below 0.001: equivalent | +0.24, 0.128: not shown |
| 104 | +0.23, 0.128: not shown | +0.05, below 0.001: equivalent |

**PL0, placebo known from planning** (the committed hybrid's seed sets; computed before the plan on committed losses):
equivalent for all four seed sets at 15 minutes (Holm p 0.006 or below) and for one of four at 30 minutes (seed set
104, Holm p 0.004; the others 0.167, 0.266 and 0.232).

* **What the placebos say.** Two versions of the **same** model, differing only in their LightGBM seeds, are called
  equivalent within 0.5 point in only 10 of 16 comparisons on one seed set each. The TOST at 0.5 point on these 792
  days has limited power, as the plan warned. The primary sample (five-seed average) removes most of the seed noise
  and still does not pass at 30 minutes, where the estimate (0.25) is well inside the margin but its 90% interval
  reaches 0.69.

## 5. Expiry days, the last hour, the open (REAL DATA, Nifty)

**S2, expiry days only** (169 days, five-seed average, Holm over 6; `tests.csv`, `absorption.csv`):

| Test | 15 min | 30 min |
|---|---|---|
| H1: `hybrid_cpp` beats `mcs_cpp` (G) | +4.83, Holm p 0.017: supported | +6.03, Holm below 0.001: supported |
| TOST (E) | +2.05, Holm p 1.000: not shown | +1.37, Holm 1.000: not shown |
| difference (E) | +2.05, stat 2.73, raw p 0.007, **Holm 0.028: different** (all five seed sets agree) | +1.37, raw 0.076, Holm 0.228: no significant difference |
| B on expiry days | 3.93 | 2.01 |
| A on expiry days [95%] | 0.48 [0.31, 0.73] | 0.32 [-0.46, 1.08] |

**E2, non-expiry days** (623 days, exploratory, raw p): G 5.14 and 7.44 (p below 0.001); E +0.17 (p 0.556) and
-0.12 (p 0.602); TOST raw p 0.125 and 0.058.

* On expiry days the committed hybrid's gain over `mcs_cpp` is only 2.78% at 15 minutes and 4.66% at 30 minutes
  (`losses.csv`); `hybrid_cpp`'s is 4.83% and 6.03%. So on expiry days the correction recovers only about half of
  our expiry-day curve's damage (A 0.48 at 15 minutes). On the other days the bases differ less (B 0.61 and 0.78) and
  the two hybrids are almost the same.
* **By expiry regime** (descriptive, 15 minutes, `losses.csv`, column `points_vs_hybrid_of_mcs_cpp`): E is +0.83
  points with several weekly expiries (312 days), -0.21 with one Thursday expiry (192 days) and +1.11 with one Tuesday
  expiry (288 days); at 30 minutes +0.50, -0.44 and +0.57. The regime where our base was least worse than the
  paper's (Thursday expiries; B 0.53 and 0.14 there, P116) is the only one where the committed hybrid is ahead.

**S1, the last hour** (14:30 forecast, seed set 7, Holm over 3): `hybrid_cpp` beats `mcs_cpp` by 5.82% (Holm p below
0.001: supported); E is -0.34 points (the committed hybrid slightly better here), TOST Holm p 0.809 (not shown),
difference Holm p 0.809 (no significant difference). The two bases hardly differ in the last hour (B 0.22), so there
was little to absorb; the TOST interval is wide (smallest passing margin 1.41) because there is one forecast a day.

**E3, the 09:15 anchor and the rest** (seed set 7, exploratory, raw p): at 15 minutes E is +3.41 points at 09:15 (p
0.055) and +0.24 at the other anchors (p 0.271); at 30 minutes -0.66 at 09:15 (p 0.715) and +0.46 elsewhere (p
0.097). By anchor (`gain_by_anchor.csv`, seed set 7, 15 minutes), the largest E are at 09:15 (+3.41), 13:15 (+2.60)
and 14:45 (+1.42); at 15:00 E is only +0.15, so the base difference P116 found at 15:00 (`mcs_ours` 6.67% worse) is
absorbed by the correction there. At 14:45, where `mcs_ours` is 5.71% worse than `mcs_cpp`, the committed hybrid is
0.62% worse than `mcs_cpp` and `hybrid_cpp` 0.80% better.

## 6. P109's combination and our implementation (REAL DATA, Nifty)

* **S3** (Holm over 2): `hybrid_cpp` does **not** beat P109's combination: the combination is 0.41 points better at 15
  minutes and 0.17 points better at 30 minutes (one-sided p 0.643 and 0.561, Holm 1.000). P109 found the same for
  the committed hybrid. Building the hybrid on the paper's model narrows the gap to the combination (from 1.06 and
  0.42 points to 0.41 and 0.17) but does not close it.
* **E4** (exploratory): `hybrid_cpp` beats `mcs_ours` by 6.42%, 8.09% and 6.02% (15 minutes, 30 minutes, last hour;
  p below 0.001).

## 7. Every mcsGARCH input as the paper specifies it (E1, exploratory, REAL DATA, Nifty, seed set 7)

`hybrid_cpp_all` also takes `log_s_window`, `log_mcs_iv` and `log_implied_window` from mcsGARCH as the paper
specifies it (`tests.csv`, raw p):

| | 15 min | 30 min | last hour |
|---|---|---|---|
| G of `hybrid_cpp_all` (beats `mcs_cpp`) | +5.30 (p below 0.001) | +7.14 (p below 0.001) | +5.85 (p below 0.001) |
| against `hybrid_cpp` (seed set 7), points | +0.35 (p 0.001) | +0.13 (p 0.343) | +0.03 (p 0.927) |
| against the committed hybrid (seed set 7), points | +0.87 (p 0.007) | +0.49 (p 0.175) | -0.31 (p 0.621) |

At 15 minutes, swapping the other mcsGARCH inputs too adds a further 0.35 points (exploratory, one seed set, not
adjusted for the 9 rows of E1). The plan expected `hybrid_cpp_all` to be close to `hybrid_cpp`; it is at 30 minutes
and in the last hour, but not quite at 15 minutes.

## 8. Bank Nifty and Fin Nifty (REAL DATA, secondary, nothing retuned)

S4 (Bank Nifty, 660 test days to 22 April 2026) and S5 (Fin Nifty, 604 test days to 27 January 2026), five-seed
average, Holm over 6 rows each; the decision rule applied descriptively (`verdict.csv`, `tests.csv`,
`absorption.csv`, `equivalence_bounds.csv`):

| Index, target | G: `hybrid_cpp` vs `mcs_cpp` (H1 Holm p) | committed hybrid vs `mcs_cpp` | E [95%] | TOST Holm p (smallest passing margin) | difference Holm p | B | A [95%] | outcome |
|---|---|---|---|---|---|---|---|---|
| Bank Nifty, 15 min | 5.01 (below 0.001) | 4.90 | +0.11 [-0.22, 0.45] | **0.028: equivalent** (0.37) | 0.471 | -0.84 | 1.13 [0.82, 2.45] | correction absorbs the base |
| Bank Nifty, 30 min | 6.16 (below 0.001) | 6.42 | -0.26 [-0.66, 0.14] | 0.368: not shown (0.60) | 0.408 | -1.23 | 0.79 [0.51, 1.20] | inconclusive |
| Fin Nifty, 15 min | 4.47 (below 0.001) | 3.97 | +0.50 [0.13, 0.88] | 1.000: not shown (0.84) | 0.057 | 0.91 | 0.45 [-1.26, 0.91] | inconclusive |
| Fin Nifty, 30 min | 5.69 (below 0.001) | 5.13 | +0.55 [-0.01, 1.10] | 1.000: not shown (1.00) | 0.128 | 0.97 | 0.43 [-2.59, 2.25] | inconclusive |

* **Descriptive verdicts**: Bank Nifty "mixed: correction absorbs the base at 15 minutes; inconclusive at 30
  minutes"; Fin Nifty "inconclusive". H1 holds on both indices at both horizons, every seed set agreeing.
* **Reading A on Bank Nifty.** There B is negative: `mcs_cpp` is the worse base, by 0.84 and 1.23 points. A = 1.13 at
  15 minutes means the hybrid built on the worse base ends up slightly better (the gap is more than absorbed); A =
  0.79 at 30 minutes means about a fifth of the base gap is left (E -0.26). Each hybrid against its own base: on Bank
  Nifty `hybrid_cpp` gains 0.92 and 0.90 points more over `mcs_cpp` than the committed hybrid gains over `mcs_ours`
  (95% intervals 0.50 to 1.41 and 0.41 to 1.45); on Fin Nifty 0.37 and 0.36 points less (-1.03 to 0.36 and -1.30 to
  0.62). So on all three indices the correction makes up more when its base is worse (on Fin Nifty the intervals
  include zero).
* **Seed sets** (`seed_repeats.csv`): Bank Nifty E is +0.05 to +0.26 at 15 minutes (all five inside 0.5); at 30
  minutes +0.14 for seed set 7 and -0.28 to -0.49 for the other four. Fin Nifty E is +0.41 to +0.59 at 15 minutes and
  +0.48 to +0.64 at 30 minutes (all positive; only two and one seed sets inside 0.5).
* **Expiry days** (descriptive, `losses.csv`, five-seed average): on Fin Nifty's 82 expiry days the committed hybrid
  is 0.97% and 0.99% *worse* than `mcs_cpp` (P116 saw the same with seed set 7), while `hybrid_cpp` is 1.01% and 1.81%
  better (E +1.98 and +2.79 points). On Bank Nifty's 84 expiry days E is +0.58 and -1.02.
* An interpretation, not a test: as on Nifty, the two hybrids differ most on expiry days, where the two bases'
  time-of-day curves differ; with 82 and 84 expiry days these numbers are uncertain.

## 9. What we expected, and what happened (REAL DATA)

| Registered expectation (plan Section 6) | Outcome |
|---|---|
| H1 holds at both horizons | **as expected** (Holm p below 0.001 at both) |
| G between the committed hybrid's gain over `mcs_cpp` (4.42, 6.83) and that plus B (about 5.9, 7.9) | **as expected**: 5.06 and 7.09 |
| E small and positive: 0 to +0.8 point at 15 minutes, 0 to +0.5 at 30 minutes | **as expected**: +0.65 and +0.25 |
| A between about 0.4 and 1 | **as expected**: 0.56 and 0.77 |
| expiry days show more of the base difference than other days | **as expected**: E +2.05 and +1.37 on expiry days, +0.17 and -0.12 on the others |
| "inconclusive" at 30 minutes; at 15 minutes "absorbs", "inconclusive" or "base matters, cpp better" | **as expected**: "inconclusive" at both |
| Bank Nifty E slightly negative | **partly**: -0.26 at 30 minutes, as expected, but +0.11 at 15 minutes (shown equivalent within 0.5 point) |
| `hybrid_cpp_all` close to `hybrid_cpp` | **partly**: within 0.13 points at 30 minutes and in the last hour, but 0.35 points better at 15 minutes (raw p 0.001, exploratory) |

## 10. What this means for how the project describes the hybrid

Under the registered wording (plan Section 7), H1 holds at both horizons and the outcome is "inconclusive":

> "Built on mcsGARCH exactly as the paper specifies it (one diurnal curve, no smoothing, ARMA(1,1) mean, Student t),
> the same hybrid recipe is 5.06% and 7.09% better than the paper's model at 15 and 30 minutes (P118, five seed
> sets; Holm p below 0.001). The committed hybrid, built on our implementation of mcsGARCH, is 4.42% and 6.83%
> better than the paper's model (five seed sets). The hybrid built on the paper's model is 0.65 and 0.25 points
> better than the committed one (95% intervals 0.05 to 1.24 and -0.30 to 0.81); the data cannot tell whether this
> difference is real, and equivalence within 0.5 point is not shown either."

* **The like-for-like figure** for the professor is `hybrid_cpp`'s: **5.06% and 7.09%** (seed set 7 alone: 4.95% and
  7.00%). It needs no caveat about our implementation of mcsGARCH, because the base and the benchmark are the same
  model.
* **The committed hybrid stays the project's model** unless Shreyan decides otherwise. Switching the project's model
  to `hybrid_cpp` after seeing these results would be a choice made on the same 792 days and should be described as
  such (plan Section 7). The difference between the two is not significant under the registered rule.
* **Do not say** "the correction absorbs the choice of base" (equivalence is not shown) or "the base does not
  matter". **Do say** that the correction absorbs about half to three quarters of the difference between the bases
  (A 0.56 and 0.77), and that what remains is on expiry days, from our separate expiry-day curve.
* **Do not say** the hybrid beats P109's combination; built on either base, it does not (S3).
* **Bank Nifty and Fin Nifty.** `hybrid_cpp` beats `mcs_cpp` on both (5.01% and 6.16% on Bank Nifty, 4.47% and
  5.69% on Fin Nifty; Holm p below 0.001). The two hybrids are shown to be equivalent only on Bank Nifty at 15
  minutes; elsewhere quote both hybrids' gains over `mcs_cpp` side by side (Section 8). These outcomes are
  descriptive: the decision rule is registered for Nifty.

## 11. Checks (REAL DATA unless marked SYNTHETIC)

* **Reproduction** (`reproduction_check.csv`, 269 rows, every one identical to the last bit, largest relative
  difference 0; `analyze` refuses to run otherwise). In every job of every index: the targets y, the rows (day,
  anchor, date), `mcs_ours` and the implied-level mcsGARCH equal the committed forecasts; `mcs_cpp` equals P116's
  `mcs_cpp` forecasts and is the same in every seed set's job; in the `hybrid_cpp_all` jobs the base equals
  `mcs_cpp`. The committed hybrid refitted alongside (seed set 7 on Bank Nifty and Fin Nifty, seed set 101 at 15
  minutes on Nifty, and the three phase-1 jobs on Nifty) equals the committed forecasts or the committed daily
  losses. Rows: Nifty 19,800, 19,008 and 792 forecasts; Bank Nifty 16,500 and 15,840; Fin Nifty 15,100 and 14,496.
* **The same days as the committed files.** The daily QLIKE of `mcs_ours` in P106's (Nifty) and P110's (Bank Nifty,
  Fin Nifty) seed-set files and in P109's files equals this study's on every day, so the committed hybrid's seed
  sets 101 to 104 and P109's combination sit on the same days; P116's committed daily QLIKE of `mcs_ours`, `mcs_cpp`
  and the hybrid equal this study's.
* **The registered analysis and the preview** (Section 13): the Nifty rows of `tests.csv`, `absorption.csv`,
  `losses.csv`, `seed_repeats.csv`, `equivalence_bounds.csv` and `gain_by_anchor.csv` are identical to the preview's.
* **Inputs.** Every raw file matches `data/MANIFEST.csv` (`run_info.json`, `input_sha256`): 1,891 files for Nifty,
  1,891 for Bank Nifty and 1,711 for Fin Nifty, none differing.
* **Code.** `analysis_commit` in `run_info.json` is the plan commit: no code changed between the plan and the
  analysis.
* **No look-ahead**, by construction and by test: every model is fitted on the days before its block only
  (`test_no_look_ahead`: changing every return after a block leaves that block's forecasts unchanged, SYNTHETIC).
* **Tests.** `tests/test_p118_hybrid_on_paper_mcsgarch.py` (SYNTHETIC only, 13 tests) passes; with the existing
  tests of every module this study touches or loads, 104 passed (phase 1). **Full suite** at the plan commit, run on
  3 November 2026 from 12:24:41 to 12:32:25 UTC (during the Nifty jobs): 697 passed, 0 failed, in 462.66 s; the only
  warnings are three existing scipy `RuntimeWarning`s in `tests/test_sensex_confound.py`. Every new or changed
  Python file compiles under Python 3.11 (`python3.11 -m py_compile`); this commit changes no Python file.

## 12. Run time

On 3 November 2026, 2 worker processes (`run_stdout_*.txt`, `phase2_launcher.log`, `timing.csv`, `run_info.json`):

| Stage | Wall time (UTC) | Loading the market, building the datasets | Jobs | Job time added up |
|---|---|---|---|---|
| Phase-1 check (Nifty, before the plan commit; jobs only) | 12:00:32 to 12:05:54 | 62 s, 189 s (plan Section 11) | 3 | 519.2 s |
| Nifty | 30.1 min, 12:18:00 to 12:48:06 | 61 s, 193 s | 14 (10 `hybrid_cpp`, 1 with the committed hybrid alongside, 3 `hybrid_cpp_all`) | 3,058.0 s |
| Bank Nifty | 15.1 min, 12:48:06 to 13:03:14 | 55 s, 145 s | 10 | 1,506.8 s |
| Fin Nifty | 12.7 min, 13:03:14 to 13:15:57 | 58 s, 137 s | 10 | 1,247.5 s |
| Analysis | 13:16:50 to 13:17:08 | | | 16.7 s |

* A Nifty job with one hybrid took 163 to 217 seconds, except two that ran while the full test suite was running
  (seed set 101: 333 s at 30 minutes, and 503 s at 15 minutes for the job that also refits the committed hybrid).
* The plan expected about 1 hour in all (Nifty 25 to 30 minutes, the others 15 to 20 minutes each); the registered
  runs took 58 minutes (12:18:00 to 13:15:57 UTC) plus 17 seconds of analysis, far inside the 18:00 UTC deadline.

## 13. Deviations from the plan

1. **None in the design.** Every registered job, test, family and output was run as written, on all three indices,
   with no change to the code after the plan commit (`analysis_commit` in `run_info.json` is the plan commit).
   Nothing was dropped, no run was restarted and no block was recomputed. Every seed set finished long before the
   18:00 UTC deadline, so no fallback rule of the plan (Section 10) was used.
2. **Commands.** The plan's commands set `PYTHONPATH=src`; the run used the absolute path of the same `src` folder
   (a sandbox rule). The three `run` commands were run one after the other by one launcher script, started with
   `setsid nohup` so that it could outlive the agent's session; it wrote START and EXIT lines to
   `phase2_launcher.log`. Same commands, arguments and order as the plan, with
   `--plan-commit 49d19ce2779451e13cbf868d325a651c4da6a1f4` for `analyze`.
3. **A preview of the analysis on Nifty only, before the other indices had run.** After the Nifty jobs finished
   (12:48 UTC), and while Bank Nifty and Fin Nifty had not yet run, the AI agent ran the registered `analyze` command
   with its output sent to a scratch folder outside the repository, to start writing this document. It used the
   same code and settings and saw the Nifty results (including the primary verdict). **Nothing in the code, the
   design, the commands or the order of the runs was changed after it**; Bank Nifty and Fin Nifty then ran as
   planned. The registered analysis (13:16:50 to 13:17:08 UTC) wrote the files in `results/`; its Nifty numbers are
   identical to the preview's in every row (Section 11). The preview is not a registered output and is not
   committed.
4. **The full test suite ran in parallel with the Nifty jobs** (12:24:41 to 12:32:25 UTC), so that it was done
   before this commit. It slowed two jobs (Section 12) and changes no number.
5. **Reading notes, not deviations.**
   * The S3 rows (`hybrid_cpp` against the combination) carry no pass or fail label in `tests_detail.csv` (the
     labels cover the H1, TOST and difference rows); they are read by their Holm p (1.000 for both).
   * `timing.csv` also contains the three phase-1 jobs (kind `repro`), and `run_info.json`'s stage-1 times for Nifty
     start with them.
   * At Nifty 15 minutes the raw S0 p (0.040) and the bootstrap interval of E (0.05 to 1.24) point to a difference
     that the registered Holm p (0.079) does not confirm. The registered rule is followed and no difference is
     claimed.
   * On Bank Nifty B is negative, so A above 1 means the hybrid built on the worse base ended slightly better.
   * The decision rule's outcomes for Bank Nifty and Fin Nifty are descriptive, as registered.

## 14. Caveats

1. **Same days.** The 792 Nifty test days are the ones on which the headline and P116 were found; this study changes
   the hybrid's base, not the data. Only later days can confirm it.
2. **Limited power for equivalence**, as the plan said (its Prior knowledge and the placebos of Section 4). "Inconclusive"
   means the data cannot tell, not that the base matters.
3. **"As the paper specifies"** carries P116's caveats: the session grid (375 minutes, no overnight gap in the first
   return), the two-step ARMA estimation, the constant in the daily AR(1) and the cross-day recursion are our
   choices; the paper's software settings are not published.
4. **`hybrid_cpp` keeps three inputs from our implementation** (`log_s_window`, `log_mcs_iv`, `log_implied_window`),
   as registered: only the base was changed. `hybrid_cpp_all` (one seed set, exploratory) suggests that swapping
   them too may help a little at 15 minutes.
5. **Seed averaging.** The five-seed average removes most of LightGBM's randomness from the primary sample, but the
   DM, TOST and bootstrap do not include the remaining seed noise; the seed rule guards against a result carried by
   one seed set. All five seed sets agreed in sign on every primary row.
6. **No tuning.** Both hybrids use the committed, untuned settings. A hybrid tuned for `mcs_cpp` might do better or
   worse.
7. **Several tests per question.** The verdict is the registered rule's; the many secondary, robustness, placebo
   and exploratory rows describe it and cannot change it. Exploratory p-values are not adjusted.

## 15. Files

All in `results/p118_hybrid_on_paper_mcsgarch/` (REAL DATA):

* `tests.csv`: every p-value (75 rows), with the registered columns; `tests_detail.csv` adds identifiers,
  samples, standard errors, 90% intervals, the smallest passing margins, the seed-rule counts and each row's label.
* `verdict.csv`: the registered verdict for Nifty (and the rule applied descriptively to Bank Nifty and Fin Nifty),
  with G, E, B, A and their intervals and the Holm p of every primary and S0 row.
* `losses.csv`: every model's mean daily QLIKE by index, target and subset (all days, expiry and non-expiry days,
  regimes), with gains over `mcs_cpp` and `mcs_ours` and the difference from the committed hybrid in points.
* `absorption.csv`: G, E, B, A and each hybrid's gain over its own base, with 95% bootstrap intervals, by index,
  target, subset and seed sample.
* `seed_repeats.csv`: every seed set's G, the committed hybrid's gain and E.
* `equivalence_bounds.csv`: every TOST row's estimate, standard error, 90% interval and smallest passing margin.
* `gain_by_anchor.csv`: every model's QLIKE and gains at every anchor minute (Nifty, seed set 7).
* `daily_losses_<index>_<target>.csv`: every model's daily QLIKE (each seed set, the five-seed averages,
  `hybrid_cpp_all` and the combination), from which every number can be recomputed.
* `reproduction_check.csv`, `timing.csv`, `params_by_job.csv`, `run_info.json`: checks, job timings, every refit's
  mcsGARCH parameters, plan commit, settings, input hashes and versions.
* `run_stdout_nifty.txt`, `run_stdout_banknifty.txt`, `run_stdout_finnifty.txt`, `phase2_launcher.log`,
  `analyze_stdout.txt`: logs.

The forecasts of every job are in `data/processed/p118/` (git-ignored).

## 16. How to reproduce

From the repository root, with the raw data listed in `data/MANIFEST.csv` in `data/raw` and P116's block files in
`data/processed/p116_ref/` (plan Section 14):

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py run --index NIFTY --phase repro --workers 2
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py run --index NIFTY --workers 2 \
    > results/p118_hybrid_on_paper_mcsgarch/run_stdout_nifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py run --index BANKNIFTY --workers 2 \
    > results/p118_hybrid_on_paper_mcsgarch/run_stdout_banknifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py run --index FINNIFTY --workers 2 \
    > results/p118_hybrid_on_paper_mcsgarch/run_stdout_finnifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p118_hybrid_on_paper_mcsgarch.py analyze \
    --plan-commit 49d19ce2779451e13cbf868d325a651c4da6a1f4 > results/p118_hybrid_on_paper_mcsgarch/analyze_stdout.txt 2>&1
```

About 1 hour on 2 cores (the phase-1 check about 6 minutes, Nifty 30, Bank Nifty 15, Fin Nifty 13; the analysis
17 seconds). The synthetic tests: `PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest
tests/test_p118_hybrid_on_paper_mcsgarch.py`. A synthetic end-to-end run of all stages: add `--synthetic` to each
command (labelled SYNTHETIC).
