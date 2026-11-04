# One pooled test of the forecasting hybrid over three indices, and how many independent indices they are worth (P107 results)

**Verdict.** Pooled over Nifty, Bank Nifty and Fin Nifty on their 604 common days, with standard errors that allow
the three indices to move together, the hybrid beats mcsGARCH by **4.7% at 15 minutes and 6.0% at 30 minutes**
(pooled t statistics 10.7 and 11.3, Holm-adjusted p 7e-25 and 5e-27). A joint bootstrap agrees. The three indices'
gains are statistically indistinguishable from each other at both horizons. **But the three indices are not three
independent pieces of evidence: together they are worth about 1.8 independent indices at 15 minutes** (95% interval
1.6 to 2.1), and about 1.8 to 2.4 at 30 minutes depending on how a few extreme days are treated. Bank Nifty adds a
good part of an independent index beyond Nifty; **Fin Nifty adds almost nothing once Nifty and Bank Nifty are in.**

*Note added 3 November 2026 after study P116:* here "mcsGARCH" means our implementation (a separate time-of-day
curve for expiry days, a 5-minute smoothing of the curves, a zero mean), not the paper's own specification. Against
the paper's specification (one curve, no smoothing, ARMA(1,1) mean) see `docs/P116_BENCHMARK_AS_PRINTED.md`, which
did not pool the indices; there the hybrid's gain at 15 and 30 minutes is 4.43% and 6.64% on Nifty, 4.73% and 6.20%
on Bank Nifty and 4.03% and 5.18% on Fin Nifty, each on its own test days rather than the 604 common days used here.

**This is a summary of evidence already seen, not new evidence.** Each index's own result was known before this study
was planned, and the plan showed in advance that the primary test could not fail (Section 3). What the registration
fixed was only how to pool. The new information is the dependence between the indices, the effective number of
indices, the test that the gains are equal, and the pooled comparison with `mcsgarch_iv`.

**REAL DATA.** Every number comes from the committed forecast files of the three pipeline runs (Nifty
`results/pipeline_nse_v2/`, Bank Nifty P91 `results/pipeline_nse_banknifty/`, Fin Nifty P104
`results/pipeline_nse_finnifty/`); nothing was refitted. Plan: `docs/prereg/P107_POOLED_THREE_INDEX.md`, committed as
eb8580a ("P107 registered plan") before any pooled statistic, correlation, effective number or bootstrap was computed.
Results: `results/p107_pooled_test/` (every p-value is in `tests.csv`). Run on 3 November 2026. The study (plan, code,
run and this write-up) was done by an AI agent (Claude) working for Shreyan Thanki.

## 1. The answer in plain words

* **The pooled test passes, as it had to.** The hybrid's average loss is 4.7% lower than mcsGARCH's at 15 minutes and
  6.0% lower at 30 minutes, averaged over the three indices. The pooled t statistics (10.7 and 11.3) are bigger than
  Nifty's alone on the same days (9.4 and 7.6), but much smaller than they would be if the three indices were
  independent (13.9 and 12.6).
* **The three gains look alike.** The test that the three indices have the same gain does not reject (p 0.23 at 15
  minutes, 0.37 at 30 minutes). Not rejecting is not proof of equality, but there is no sign that one index is
  different at these horizons.
* **How much do Bank Nifty and Fin Nifty add?** On a given day, the hybrid's advantage on Bank Nifty is only weakly
  related to its advantage on Nifty (correlation 0.22 at 15 minutes), so Bank Nifty is close to an independent check.
  Fin Nifty's daily advantage is more strongly related to Bank Nifty's (0.60) and to Nifty's (0.41), so once Nifty and
  Bank Nifty are counted, Fin Nifty adds almost nothing (-0.05 and +0.01 of an index at 15 and 30 minutes). In short: **three
  related indices are worth about two independent ones.**
* **Other pooled results.** The last hour: 5.0% better (supported). Each index's own expiry days: 3.7% and 4.5% better
  at 15 and 30 minutes (supported; Fin Nifty alone was not significant). Against `mcsgarch_iv` (mcsGARCH whose daily
  level comes from option prices, an information-matched benchmark): 2.5% and 3.0% better, Holm p 0.043 at both
  horizons, so supported, but only just. On Bank Nifty alone the hybrid's edge over `mcsgarch_iv` is not significant.
* **In the last hour the three indices differ clearly** (exploratory, p 3e-5): Fin Nifty's last-hour gain is near zero.
  On expiry days the pooled last-hour gain is zero (+0.2%, worse).

## 2. Two ideas, explained simply

**Driscoll-Kraay standard errors.** A test asks whether an average is far from zero compared with its noise. If you
average 1,812 numbers (3 indices x 604 days) and treat them as 1,812 independent draws, you underestimate the noise,
because the three indices share the same day's news, and neighbouring days are related. Driscoll and Kraay (1998)
fix both problems with a simple recipe. First, on each day, take the average of the three indices' results, so that
one day counts once, however the indices move together. Then measure how noisy that daily series is over time,
allowing neighbouring days to be related (the Newey-West correction the project already uses for one index). With one
index this is exactly the project's usual Diebold-Mariano test. It needs many days, not many indices, which suits 604
days and 3 indices.

**Effective number of indices.** Suppose you had three indices whose daily results were completely unrelated. Then
averaging them would cut the noise variance by a factor of 3: you would have "3 indices' worth" of evidence. If they
were identical copies, averaging would not help at all: "1 index's worth". The effective number N_eff measures where
the truth lies. The registered headline is the most direct version: N_eff = 3 x (noise variance of the pooled average
if the indices were independent) / (its actual noise variance, from Driscoll-Kraay). With equally noisy indices it is
3 / (1 + 2 x their average correlation). It can exceed 3 if the correlations are negative.

## 3. What was known in advance (the registered bounds)

Each index's own test on the 604 common days was already committed (`results/pipeline_nse_finnifty/p104/other_indices_same_dates.csv`).
From those numbers alone the plan computed (`results/p107_pooled_test/phase1/bounds_from_committed.csv`) that the
pooled t statistic must be at least a floor (its value if the indices were perfectly correlated), and would reach a
higher value if the indices were independent (it could go above that only if their results were negatively
correlated). The observed values (REAL DATA, `tests.csv`) are above the floor, as they had to be, and below the
independent value:

| Target, 604 days | t Nifty | t Bank Nifty | t Fin Nifty | floor (perfectly correlated) | **observed pooled t** | if independent | N_eff implied |
| --- | --- | --- | --- | --- | --- | --- | --- |
| next 15 minutes | 9.40 | 8.06 | 7.01 | 8.07 | **10.71** | 13.87 | 1.79 |
| next 30 minutes | 7.61 | 7.27 | 7.07 | 7.32 | **11.31** | 12.62 | 2.41 |
| last hour | 2.13 | 4.84 | 0.52 | 2.44 | **3.85** | 4.14 | 2.60 |

The last column is the design-effect N_eff of `effective_n.csv` (it equals 3 x (observed / independent)^2 up to the
small-sample factor).

## 4. Method (short)

* **Sample.** The 604 dates present in all nine forecast files, 10 August 2023 to 27 January 2026 (all of Fin Nifty's
  test days). On every date the three indices have the same forecast times (25 a day at 15 minutes, 24 at 30 minutes,
  one last-hour window). Expiry days: each index's own (Nifty 130, Bank Nifty 82, Fin Nifty 82; 294 index-days on 252
  dates).
* **Daily differences.** For each index and day: the day's mean QLIKE loss of mcsGARCH minus that of the hybrid
  (`hybrid_base`), positive when the hybrid is better; exactly the series the project's DM tests use. Our per-index
  numbers match P104's committed table on all 18 rows (`check_against_p104.csv`).
* **Pooled estimate.** The equally weighted average of the three indices' mean differences (registered: raw QLIKE,
  which is unit-free; robustness R1 divides each index by its own mcsGARCH loss).
* **Test.** Driscoll-Kraay covariance of the three means (OLS on index dummies), Bartlett kernel, lag
  floor(4 (T/100)^(2/9)) = 5 for 604 dates (4 for the 252 expiry dates), Harvey-Leybourne-Newbold factor, Student t
  with T - 1 degrees of freedom; one-sided. Holm over the two horizons within each family.
* **Bootstrap.** Stationary bootstrap over dates, all three indices resampled together, mean block 10 days, 9,999
  draws, seed `[107, 10]`.
* **Code.** Our own numpy implementation, checked on SYNTHETIC data against a hand calculation, against statsmodels'
  `hac-groupsum` (balanced and unbalanced panels), against the project's DM test for one index, and for its size under
  strong cross-sectional dependence (22 tests in `tests/test_p107_pooled_test.py`).

## 5. The primary test, P1 (REAL DATA)

| Horizon | Nifty mean diff | Bank Nifty | Fin Nifty | pooled diff | pooled change | DK s.e. | DK t | raw p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next 15 minutes | 0.01364 | 0.01056 | 0.01251 | 0.01224 | -4.75% | 0.00114 | 10.71 | 6.6e-25 | 6.6e-25 |
| next 30 minutes | 0.01418 | 0.01063 | 0.01240 | 0.01241 | -6.03% | 0.00110 | 11.31 | 2.4e-27 | 4.9e-27 |

(Mean daily QLIKE difference, mcsGARCH minus hybrid; "pooled change" = -100 x sum of the three differences / sum of
the three mcsGARCH mean losses. Sources: `tests.csv` rows P1, `pooled_estimates.csv`.) **Verdict: pooled gain
confirmed** (`verdicts.csv`).

How much the dependence matters (`se_comparison.csv`): the Driscoll-Kraay variance of the pooled mean is 1.68 times
(15 minutes) and 1.25 times (30 minutes) what it would be if the indices were independent. A naive test that treated
all 1,812 index-days as independent would report t of 14.1 and 12.8 instead of 10.7 and 11.3.

## 6. Secondary hypotheses (REAL DATA, each reported whatever its direction)

| Family | What | 15 minutes | 30 minutes | Verdict |
| --- | --- | --- | --- | --- |
| S1a | joint bootstrap (block 10, 9,999 draws) | p 0.0001 (Holm 0.0002) | p 0.0001 (Holm 0.0002) | **bootstrap agrees** |
| S1b | three gains equal? (Wald, F(2, 603)) | F 1.47, p 0.23 (Holm 0.46) | F 1.00, p 0.37 (Holm 0.46) | **no evidence that the gains differ** |
| S1d | pooled, each index's own expiry days | -3.7%, t 3.25, p 0.0007 (Holm 0.0013) | -4.5%, t 3.03, p 0.0013 (Holm 0.0013) | **supported** |
| S1e | pooled, against `mcsgarch_iv` | -2.5%, t 2.02, p 0.022 (Holm 0.043) | -3.0%, t 2.03, p 0.021 (Holm 0.043) | **supported** (narrowly) |

| Family | What | Result | Verdict |
| --- | --- | --- | --- |
| S1c | pooled last hour (14:30 to 15:30) | -5.0%, t 3.85, p 6.5e-5 | **supported** (certain in advance, Section 3) |

Notes.

* **S1a.** 0.0001 is the smallest p the bootstrap can give (1 / 10,000): no draw came near zero. The bootstrap's
  standard deviation of the pooled mean (0.00114 and 0.00107) is almost the Driscoll-Kraay standard error (0.00114 and
  0.00110), which is a good independent check of the DK calculation. 95% bootstrap interval of the pooled change: -5.6%
  to -3.9% at 15 minutes, -7.0% to -5.1% at 30 minutes (`bootstrap.csv`).
* **S1b.** The three mean differences are 0.0136, 0.0106 and 0.0125 at 15 minutes (Nifty, Bank Nifty, Fin Nifty) and
  0.0142, 0.0106 and 0.0124 at 30 minutes. Bank Nifty's is the smallest, but the differences are within noise.
* **S1d.** Each index's mean is over its own expiry days and the three means are equally weighted (the time axis is the
  252 dates on which at least one index expires). With every expiry index-day weighted equally instead (R4), t is 3.59
  and 3.23. Per index the gains were 4.1%, 2.9% and 3.9% at 15 minutes (only Fin Nifty was not significant on its own).
* **S1e.** Per index (exploratory E2, one-sided DM p): Nifty 3.6% (p 0.027) and 4.2% (p 0.014); Bank Nifty 1.4%
  (p 0.095) and 1.3% (p 0.24); Fin Nifty 2.4% (p 0.047) and 3.4% (p 0.034). The pooled edge over the
  information-matched benchmark is about half the edge over plain mcsGARCH, and on Bank Nifty it is not significant.
  The daily differences against `mcsgarch_iv` are more correlated across indices (Pearson 0.50 to 0.67 at 15 minutes),
  so here the three indices are worth only about 1.5 independent ones (`effective_n.csv`).

## 7. Correlations and the effective number of indices (REAL DATA, descriptive)

**Correlation of the daily loss differences** (mcsGARCH minus hybrid) across indices, with 95% bootstrap intervals
(`correlations.csv`):

| Pair | 15 min Pearson | 15 min Spearman | 15 min long-run | 30 min Pearson | 30 min Spearman | last hour Pearson | last hour Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Nifty, Bank Nifty | 0.22 (-0.01 to 0.42) | 0.30 | 0.09 (-0.13 to 0.34) | -0.03 (-0.35 to 0.34) | 0.23 | 0.02 (-0.11 to 0.20) | 0.09 |
| Nifty, Fin Nifty | 0.41 (0.34 to 0.50) | 0.44 | 0.41 (0.30 to 0.53) | 0.22 (-0.14 to 0.62) | 0.45 | 0.06 (-0.14 to 0.30) | 0.20 |
| Bank Nifty, Fin Nifty | 0.60 (0.43 to 0.73) | 0.57 | 0.49 (0.27 to 0.70) | 0.38 (-0.01 to 0.71) | 0.51 | 0.45 (0.33 to 0.60) | 0.58 |

For comparison, the daily mcsGARCH **loss levels** are much more correlated (15 minutes: 0.62, 0.66 and 0.88): when
volatility surprises mcsGARCH on one index it surprises it on the others, but whether the hybrid's correction helps
that day is far more index-specific. (The indices' returns themselves are far more alike: Nifty's and Bank Nifty's last-30-minute
returns correlate 0.833, `docs/FALSIFICATION.md` T5.)

**Effective number of indices** (`effective_n.csv`; against mcsGARCH; 95% bootstrap intervals):

| Measure | 15 minutes | 30 minutes | last hour |
| --- | --- | --- | --- |
| **design effect of the pooled test (registered headline)** | **1.79 (1.56 to 2.10)** | **2.41 (1.60 to 3.56)** | **2.60 (1.82 to 3.61)** |
| same, from bootstrap variances | 1.73 | 2.45 | (not computed) |
| Kish, Pearson correlations | 1.65 (1.48 to 1.90) | 2.18 (1.50 to 3.16) | 2.22 (1.80 to 2.71) |
| Kish, long-run correlations | 1.81 (1.53 to 2.14) | 2.41 (1.61 to 3.46) | 2.36 (1.80 to 3.03) |
| eigenvalue participation ratio | 2.16 (1.91 to 2.40) | 2.66 (1.94 to 2.88) | 2.64 (2.35 to 2.78) |
| Nyholt (2004) | 2.61 (2.43 to 2.75) | 2.87 (2.45 to 2.96) | 2.86 (2.72 to 2.92) |
| Li and Ji (2005) | 3.00 | 3.00 | 3.00 |
| eigenvalues of the correlation matrix | 1.84, 0.80, 0.36 | 1.42, 1.03, 0.55 | 1.45, 0.99, 0.55 |
| "Nifty equivalents", equal weights | 1.61 (0.88 to 2.61) | 2.89 (1.14 to 5.85) | 4.18 (1.74 to 7.48) |
| "Nifty equivalents", best (GLS) weights | 2.05 (1.51 to 3.23) | 3.27 (1.70 to 7.19) | 4.33 (1.85 to 9.35) |

How to read it. The design effect is the one tied to the test: at 15 minutes the pooled test has the precision of
1.8 independent indices. The eigenvalue-based measures (participation ratio, Nyholt, Li and Ji) are built for counting
tests in genetics and are coarse with only three series (Li and Ji gives exactly 3 whenever the largest eigenvalue is
below 2, as here); they are reported because the plan named them, not because they answer this question better. "Nifty
equivalents" say how much more precise the pooled average is than Nifty's own average: 1.6 times at 15 minutes. They
can exceed 3 when another index is less noisy than Nifty, as Bank Nifty is in the last hour.

**What each index adds** (design effect of all three minus that of the pair without it):

| Index added last | 15 minutes | 30 minutes | last hour |
| --- | --- | --- | --- |
| Nifty | +0.43 | +0.91 | +1.30 |
| Bank Nifty | +0.36 | +0.75 | +0.59 |
| Fin Nifty | **-0.05** | **+0.01** | +0.32 |

Pairs alone (design effect): Nifty and Bank Nifty 1.84 and 2.40 (15 and 30 minutes), Nifty and Fin Nifty 1.43 and
1.66, Bank Nifty and Fin Nifty 1.36 and 1.50. At 15 minutes the Nifty plus Bank Nifty pair is even more precise than
all three with equal weights ("Nifty equivalents" 2.02 against 1.61), because Fin Nifty is the noisiest of the three
and moves with the other two.

**Placebo (registered, descriptive).** Shifting Bank Nifty's series by 21 trading days and Fin Nifty's by 42 breaks
the shared dates. The correlations then fall to between -0.03 and 0.03 at 15 and 30 minutes (-0.08 to 0.01 in the last
hour) and the design effect rises to 3.07, 3.03 and 3.15. So the dependence measured above comes from the indices
sharing days, as it should.

## 8. Robustness, exploratory and post hoc results (REAL DATA)

**Robustness of P1** (`tests.csv`): scaled differences (R1, the average proportional gain over the three indices):
4.74% (t 10.67) and 6.02% (t 11.30). Twice the bandwidth (R2, 10 lags): t 10.94 and 11.59. Bootstrap with mean blocks
of 5 and 20 days (R3): p 0.0001 in all four cases. Nothing changes.

**Exploratory** (`tests.csv`, role exploratory; no verdicts):

* **E1, Bank Nifty and Fin Nifty only** (the two indices the hybrid was not designed on): 4.55% better at 15 minutes
  (t 8.59, p 4e-17) and 5.70% at 30 minutes (t 8.74, p 1e-17) (`stdout.txt`, `tests.csv`). The out-of-sample indices alone carry a very strong pooled result.
* **E2, each index against `mcsgarch_iv`**: Section 6, S1e notes; in the last hour Nifty 8.0% (p 0.023), Bank Nifty 3.6%
  (p 0.20), Fin Nifty 4.0% (p 0.22).
* **E3, pooled last hour against `mcsgarch_iv`**: 5.3% better, t 1.47, p 0.071.
* **E4, pooled last hour on expiry days**: +0.2% (hybrid slightly worse), t -0.06, p 0.53.
* **E5, are the three last-hour gains equal?** No: F 10.5, p 3.3e-5. The last-hour gains are 5.4% (Nifty), 8.2%
  (Bank Nifty) and 1.0% (Fin Nifty).

**Post hoc: a few extreme days** (not in the plan; written after the results were seen; descriptive, no p-values;
`experiments/p107_posthoc.py`, `posthoc_influence.csv`, `posthoc_extreme_days.csv`). At 30 minutes the Pearson
correlations were far below the Spearman ones and had wide intervals. The cause is a handful of days on which the
indices disagreed sharply: on 4 June 2024 (election results) the hybrid's 30-minute advantage was +0.735 on Nifty,
-0.004 on Bank Nifty and -0.266 on Fin Nifty; on 18 April 2024 +0.438, -0.366 and +0.292 (the average daily value is
about 0.01, `pooled_estimates.csv`). Taming those days lowers the effective number:

| Variant | 15 min design effect | 30 min design effect | last hour design effect |
| --- | --- | --- | --- |
| all days (registered) | 1.79 | 2.41 | 2.60 |
| each index winsorized at its 1st and 99th percentile | 1.76 | 1.94 | 2.09 |
| without the 3 most extreme days | 1.91 | 1.77 | 2.13 |
| without the 12 most extreme days | 1.77 | 2.11 | 2.14 |

So at 15 minutes about 1.8 is stable; at 30 minutes and in the last hour the registered values (2.4 and 2.6) are pushed
up by a few days of disagreement, and "about 2" describes a typical day better. The registered test itself uses all
days, as it should; this only changes how to describe the dependence.

## 9. Deviations from the plan

* **None in the registered analysis.** Every registered test, robustness check, exploratory test, placebo and
  descriptive measure was computed as written, on the registered sample, with the registered settings and seeds. The
  per-index check against P104 matched on all 18 rows, so the stop rule did not apply.
* **Two cosmetic code changes after the first real-data run** (3 November 2026, after the results were seen): the input
  file paths in `run_info.json` are now written relative to the repository (they were absolute paths of the agent's
  working folder), and `run_info.json` now also records the script's SHA-256 and whether it differs from the commit
  it ran on. The run was repeated; all ten result CSV files are byte-for-byte identical to the first run (checked with
  SHA-256). `run_info.json` therefore shows `code_commit_at_run` eb8580a (the plan commit) with
  `script_differs_from_code_commit` true; the committed script in the phase-2 commit is the one that wrote the files.
* **Added after the results were seen (post hoc):** `experiments/p107_posthoc.py` and its two output files (Section 8),
  and one synthetic test of its helper functions (so the test file has 22 tests, not the 21 the plan listed).

## 10. Caveats

1. **Not new evidence.** The pooled test summarizes three results already seen; P1 and S1c were certain in advance
   (Section 3). It must not be counted as a fourth replication.
2. **The Nifty part is in-sample for the design** (the hybrid was designed while looking at Nifty data up to June 2025,
   which overlaps these dates). E1 shows that the two out-of-sample indices alone give a strong pooled result.
3. **The models were fitted separately on each index.** The pooled test asks whether the hybrid *method* wins on
   average across the three indices, not whether one model wins on a portfolio of them.
4. **604 common days only.** Nifty's later days (including the closing-auction period) and Bank Nifty's later days are
   left out so that the panel is balanced.
5. **Effective numbers are noisy.** They are ratios of noisy variances from 604 heavy-tailed daily values; the 30-minute
   interval runs from 1.6 to 3.6, and a few extreme days move the point estimate (Section 8).
6. **Equal weights are a choice.** Fin Nifty is the noisiest index and counts as much as the others. With the best
   weights (GLS, assuming equal gains) the three are worth 2.05 Nifty samples at 15 minutes against 1.61 with equal
   weights.
7. **The edge over `mcsgarch_iv` is fragile.** Both Holm p-values are 0.043, just under 0.05, and Bank Nifty alone shows
   no significant edge.

## 11. What this changes in the project's claims

* **Replace "replicated on three indices" by a calibrated version:** "the hybrid beats mcsGARCH on three related NSE
  indices; pooled with Driscoll-Kraay errors, the gain is 4.7% at 15 minutes and 6.0% at 30 minutes (t 10.7 and 11.3).
  Because the indices' daily results are correlated, the three are worth about two independent replications, not
  three." Fin Nifty should be described as a robustness check that adds little independent information beyond Bank
  Nifty, not as a third independent confirmation.
* **The size of the gain can be stated as one number with an interval:** 3.9% to 5.6% at 15 minutes and 5.1% to
  7.0% at 30 minutes (95% bootstrap intervals of the pooled change), with no evidence that it differs between the indices.
* **Expiry days:** pooled over each index's own expiry days the gain is 3.7% and 4.5% and significant, which is
  consistent with the per-index results (Fin Nifty alone was not significant). This too is a summary, not new evidence.
* **Last hour:** a pooled gain of 5.0%, but the three indices clearly differ there (Fin Nifty about 1%), and on expiry
  days the pooled last-hour gain is zero. Claims about the last hour should stay index-specific.
* **Against the information-matched benchmark `mcsgarch_iv`:** the pooled edge is 2.5% and 3.0%, about half the edge over
  plain mcsGARCH, and only just significant after Holm. The claim "the 15 and 30 minute gain also holds against an
  mcsGARCH given the same information" is supported when pooled, but should be stated as modest.

## 12. Files and how to reproduce

* Plan: `docs/prereg/P107_POOLED_THREE_INDEX.md` (commit eb8580a). Phase-1 files: `results/p107_pooled_test/phase1/`
  (`counts.json`, `bounds_from_committed.csv`).
* Code: `experiments/p107_pooled_test.py` (registered), `experiments/p107_posthoc.py` (post hoc). Tests (SYNTHETIC only):
  `tests/test_p107_pooled_test.py`.
* Results (REAL DATA), `results/p107_pooled_test/`: `tests.csv` (every p-value; 35 rows), `verdicts.csv`,
  `pooled_estimates.csv`, `bootstrap.csv`, `se_comparison.csv`, `correlations.csv`, `effective_n.csv`, `per_index.csv`,
  `check_against_p104.csv`, `daily_losses.csv`, `run_info.json`, `stdout.txt`, and the post hoc
  `posthoc_influence.csv` and `posthoc_extreme_days.csv`.
* Reproduce, from the repository root (about 15 seconds on one core):

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p107_pooled_test.py count
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p107_pooled_test.py run --plan-commit eb8580af15d0ddbd6e875887440a2d73fbd663c4
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p107_posthoc.py
```
