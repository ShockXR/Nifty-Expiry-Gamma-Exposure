# P107 pre-registration: one pooled test of the forecasting hybrid over Nifty, Bank Nifty and Fin Nifty, allowing for their dependence

**Status.** Written on 3 November 2026 (phase 1) by an AI agent (Claude) working for Shreyan Thanki. **This plan is
committed to git before any pooled statistic, cross-index correlation, effective number of indices or bootstrap has
been computed on real data.** In phase 1 only calendar facts were read from the real forecast files (the columns
`date`, `minute`, `is_expiry` and `regime`; no realized variance, forecast or loss), their SHA-256 hashes were taken,
and the committed table `results/pipeline_nse_finnifty/p104/other_indices_same_dates.csv` was turned into the bounds
of Section 2 by arithmetic. The code was tested on SYNTHETIC data only. The commit that adds this file is the time
stamp; phase 2 (the run) starts only after it exists. Phase 2 writes `results/p107_pooled_test/` and
`docs/P107_POOLED_THREE_INDEX.md`. All phase-2 results will be REAL DATA.

## In plain words

The project's forecasting result is that a "hybrid" model (the anchor paper's mcsGARCH, corrected by the
machine-learning model LightGBM) forecasts the next 15 and 30 minutes of volatility better than mcsGARCH alone. It was
shown three times, each index tested on its own: Nifty (792 test days), Bank Nifty (P91, 660 days) and Fin Nifty
(P104, 604 days). But the three tests are not three independent pieces of evidence. They cover the same calendar days,
the same news, the same India VIX and the same participant file, and the three indices move together (Fin Nifty is
about one third Bank Nifty's banks). If the hybrid has a good day on Nifty, it probably has a good day on Bank Nifty
too.

This study asks two questions:

1. **What does one test over all three indices say, if it is done properly?** "Properly" means it must not pretend that
   1,812 index-days are 1,812 independent observations. We use **Driscoll-Kraay standard errors**: on every day we
   first average the three indices' results, and then measure the noise of that daily average over time, allowing
   neighbouring days to be related (as the project's usual Diebold-Mariano test does for one index). Because the
   averaging happens within each day, any correlation between the indices on the same day is automatically counted.
2. **How much independent information do three indices really give?** We measure how correlated the three indices'
   daily results are, and turn that into an **effective number of indices**: "the three indices together are worth
   as much as N_eff independent ones". N_eff near 1 means Bank Nifty and Fin Nifty add little beyond Nifty; N_eff near
   3 means they are like independent replications.

## Candour: what this study can and cannot add

* **Each index's own result is already known** (Prior knowledge, below). This registration fixes only *how* the three
  are pooled. The pooled test summarizes evidence that has already been seen; **it is not new evidence**, and it must
  not be counted as a fourth replication. This sentence will be repeated in the results document.
* **The primary test cannot fail.** For an equally weighted average of the three indices, the pooled t statistic can
  never be smaller than the smallest of the three per-index t statistics, whatever the correlation (Section 2 gives the
  argument). On the common dates those are already known: 7.01 at 15 minutes and 7.07 at 30 minutes. So the primary
  test is certain to reject; its registered job is to fix the method and report the size of the pooled statistic. The
  same argument makes the pooled last-hour test (S1c) certain to give t of at least 2.44.
* **What is genuinely unseen:** the correlation of the daily loss differentials across indices (nobody in the project
  has computed it), and therefore where the pooled statistic lands between its two known bounds; the effective number
  of indices; the test that the three gains are equal; the pooled expiry-day statistic (only approximately bounded);
  the direct comparison of the hybrid with `mcsgarch_iv` on Bank Nifty and Fin Nifty (never tested directly); and the
  bootstrap.

## Prior knowledge (what was known when this was written)

* **Per-index results on the 604 common dates** (`results/pipeline_nse_finnifty/p104/other_indices_same_dates.csv`;
  hybrid against mcsGARCH, change in mean QLIKE, one-sided DM statistic):

  | Target | Nifty | Bank Nifty | Fin Nifty |
  | --- | --- | --- | --- |
  | next 15 minutes, all 604 days | -5.1% (9.40) | -4.2% (8.06) | -4.9% (7.01) |
  | next 30 minutes, all 604 days | -6.6% (7.61) | -5.3% (7.27) | -6.1% (7.07) |
  | last hour, all 604 days | -5.4% (2.13) | -8.2% (4.84) | -1.0% (0.52) |
  | next 15 minutes, own expiry days | -4.1% (4.05), 130 days | -2.9% (2.61), 82 days | -3.9% (1.53), 82 days |
  | next 30 minutes, own expiry days | -4.4% (2.84) | -3.5% (2.36) | -5.3% (1.70) |
  | last hour, own expiry days | -5.8% (1.60) | +4.7% (-0.74) | +4.4% (-0.86) |

* **The full-sample results** of `docs/BANKNIFTY_FORECAST.md` and `docs/THIRD_INDEX_FORECAST.md`, including the
  scoreboards against mcsGARCH. From them the hybrid's edge over `mcsgarch_iv` (mcsGARCH whose daily level comes from
  option prices) can be guessed but was never tested: on Bank Nifty's 660 days the hybrid is 3.9%, 5.0% and 8.1%
  better than mcsGARCH and `mcsgarch_iv` 2.8%, 4.0% and 4.1% better (15 minutes, 30 minutes, last hour); on Fin
  Nifty's 604 days the hybrid is 4.9%, 6.1% and 1.0% better, and `mcsgarch_iv` 2.5% and 2.7% better and 3.2% *worse*
  (`scoreboard_vs_mcsgarch.csv` of P91 and P104). On Nifty (792 days) the hybrid beat
  `mcsgarch_iv` by 4.8% (p 0.005) at 15 minutes and 5.9% (p below 0.001) at 30 minutes (`docs/REFEREE_RESPONSE_FORECAST.md`, M5).
* **Return correlation:** Nifty's and Bank Nifty's last-30-minute returns correlate 0.833 (`docs/FALSIFICATION.md`,
  T5). No correlation of forecast **loss differentials** across indices has been computed in the project (searched
  `docs/`, `experiments/`, `src/` and `scripts/` for "Driscoll", "effective number" and "correlation of" differentials).
* **The Nifty hybrid is not out of sample for its own design.** Every design choice was made looking at Nifty's
  development period (to 30 June 2025), which overlaps the common dates. Bank Nifty and Fin Nifty are out of sample for
  the design. An exploratory test (E1) pools only those two.
* **Phase-1 facts** (Section 2), from `python experiments/p107_pooled_test.py count`, saved in
  `results/p107_pooled_test/phase1/counts.json` and `phase1/bounds_from_committed.csv`.

## 1. Inputs and the exact command

Inputs (REAL DATA, committed; nothing is refitted), SHA-256 in `phase1/counts.json` and again in phase 2's
`run_info.json`:

| Index | Folder | Files |
| --- | --- | --- |
| Nifty | `results/pipeline_nse_v2/` | `forecasts_next15.csv` (6e42630d...), `forecasts_next30.csv` (8a07a63c...), `forecasts_last_hour.csv` (e9010659...) |
| Bank Nifty (P91) | `results/pipeline_nse_banknifty/` | (42dd6206...), (4b8639b2...), (c0898ed0...) |
| Fin Nifty (P104) | `results/pipeline_nse_finnifty/` | (8bc9c00d...), (6f02b4b7...), (92e40b6f...) |

Columns read: `date`, `minute`, `is_expiry`, `y` (realized variance), `mcsgarch`, `hybrid_base`, `mcsgarch_iv`.

Command, from the repository root, at the commit that adds this file, one process:

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p107_pooled_test.py run --plan-commit <hash of this commit>
```

## 2. Sample (phase-1 counts, no loss computed)

* **Common dates:** the dates present in all nine files: **604**, from **10 August 2023 to 27 January 2026**. These
  are all of Fin Nifty's test days; Nifty's 792 and Bank Nifty's 660 test days contain them.
* **Anchors:** on every common date the three indices have the same forecast times: 25 a day at 15 minutes (15,100
  per index), 24 at 30 minutes (14,496) and one last-hour window (604). Checked for every date.
* **Expiry days among the common dates** (each index's own `is_expiry`): Nifty 130, Bank Nifty 82, Fin Nifty 82,
  together 294 index-days on 252 distinct dates. All three expire on 13 dates; Nifty and Bank Nifty on 25, Nifty and
  Fin Nifty on 13, Bank Nifty and Fin Nifty on 17.
* **Regimes** (pipeline labels): 312, 192 and 100 common dates.

**What the committed per-index results already imply** (`phase1/bounds_from_committed.csv`; arithmetic only). For index
i with mean daily difference mu_i and DM statistic t_i, the standard error is s_i = mu_i / t_i. For the equally weighted
pooled mean, the pooled t equals sum(mu) / sqrt(1' Omega 1), where Omega is the (Newey-West) covariance of the three
mean differences. Because Omega is positive semi-definite, sqrt(1' Omega 1) is at most sum(s), so the pooled t is at
least sum(mu)/sum(s), which in turn is at least the smallest t_i (a weighted average of the t_i). With independent
indices it would be sum(mu)/sqrt(sum(s^2)). The DK test of Section 4 uses the same bandwidth (5 lags), the same
small-sample factor and the same Student t as those per-index tests, so the bounds apply to it exactly:

| Target, all 604 days | t Nifty | t Bank Nifty | t Fin Nifty | pooled t if perfectly correlated (the floor) | pooled t if independent |
| --- | --- | --- | --- | --- | --- |
| next 15 minutes | 9.40 | 8.06 | 7.01 | 8.07 | 13.87 |
| next 30 minutes | 7.61 | 7.27 | 7.07 | 7.32 | 12.62 |
| last hour | 2.13 | 4.84 | 0.52 | 2.44 | 4.14 |

On expiry days the three indices use different days, so the same arithmetic is only approximate: about 2.25 to 3.38 at
15 minutes and 2.09 to 3.21 at 30 minutes. Where the observed pooled t falls between the two columns is the new
information: the design-effect effective number (Section 6) equals 3 times the square of (observed pooled t / pooled t
if independent).

## 3. Loss differentials

For index i, target h and date t: the project's daily mean QLIKE (`niftygex.evaluation.daily_losses`, QLIKE = y/f -
log(y/f) - 1 per forecast, averaged over the day's anchors) of each model, and

d_it = QLIKE_t(benchmark) - QLIKE_t(hybrid_base),

positive when the hybrid is better. The benchmark is `mcsgarch` except in S1e and E2 and E3 (`mcsgarch_iv`). These are
exactly the series the project's DM tests use for one index.

**Pooling scale, decided now: raw QLIKE differentials, equally weighted across indices (primary).** Reasons: QLIKE
depends only on the ratio y/f, so it is unit-free and comparable across indices with different volatility levels; the
three indices' mcsGARCH mean QLIKE on the common dates is similar (0.267, 0.251 and 0.255 at 15 minutes, from the
committed table); and the raw pooled mean is a plain linear statistic with no estimated denominator. **Robustness
(R1):** each index's d_it divided by that index's mean daily mcsGARCH QLIKE over the same sample (an average
proportional gain; the denominators are treated as fixed). The pooled change in percent is reported descriptively as
-100 x sum_i mean(d_i) / sum_i mean(mcsGARCH QLIKE_i).

## 4. The Driscoll-Kraay pooled test

**Estimator.** OLS of d_it on one dummy per index gives the three mean differences mu_i; the pooled estimate is
a' mu with a = (1/3, 1/3, 1/3). Its covariance is the Driscoll and Kraay (1998) estimator: for each date the residual
scores of the indices observed that date are summed, and a Newey-West (Bartlett) kernel is applied to that series of
date sums: V = (X'X)^-1 S (X'X)^-1, S = sum_t h_t h_t' + sum_{j=1..L} (1 - j/(L+1)) sum_t (h_t h_{t-j}' + h_{t-j} h_t').
For the balanced all-days panel this is exactly the Newey-West variance of the mean of the daily cross-sectional
average (tested). It allows any correlation between indices on the same date and between dates up to L days apart. It
needs many dates, not many indices, which suits 604 dates and 3 indices (clustering by index would have only 3
clusters and is not usable).

* **Bandwidth rule (fixed now):** L = floor(4 (T/100)^(2/9)), T = number of dates in the time axis: the rule of
  Hoechle (2007, Stata `xtscc`) and of the project's DM test. For the 604 common dates **L = 5**. Robustness R2: 2L = 10.
* **Small-sample treatment (fixed now):** as in `niftygex.evaluation.diebold_mariano`: the t statistic is multiplied by
  the Harvey-Leybourne-Newbold factor sqrt((T - 1)/T) and compared with a Student t with T - 1 degrees of freedom. With a
  single index the pooled test is then exactly the project's DM test (tested). No other small-sample factor.
* **Time axis for subsets with different day sets (expiry days):** the dates on which at least one index is in the
  subset, in calendar order (252 dates for expiry days, so L = 4, df 251); an index absent on a date contributes zero
  to that date's sum. Each index's mean is over its own expiry days and the three means are **equally weighted**, so the
  estimand is "the average over the three indices of each index's mean expiry-day gain"; this matches the all-days
  primary, where equal weight per index and per index-day coincide. Robustness R4: equal weight per expiry index-day
  (pooled OLS on a constant over the 294 index-days, same DK covariance).
* **Implementation:** our own numpy code (`index_means_dk`, `driscoll_kraay_cov`, `combo_test` in
  `experiments/p107_pooled_test.py`). statsmodels 0.15.0 offers the same estimator as `cov_type="hac-groupsum"`
  (documented as "Driscoll and Kraay"; its default `use_correction="cluster"` adds a small-sample factor, so the tests
  compare with `use_correction=False`, which its source says matches Stata's `xtscc`). The tests check our function
  against a hand calculation, against statsmodels for a regression with a regressor (balanced) and for index dummies
  (unbalanced), against the project's DM test for one index, and its size (5% nominal) under strong cross-sectional
  dependence, where a naive test that treats index-days as independent over-rejects.

## 5. Hypotheses, tests and decision rules

All p-values go to `results/p107_pooled_test/tests.csv` with columns `test_id, hypothesis, role, sample, estimate,
statistic, p_raw, sidedness, p_method, p_adjusted, adjust_method` (plus `data` = REAL DATA). One-sided tests have the
alternative "the hybrid has the lower loss" (d > 0). Holm (`niftygex.evaluation.holm`) is applied within each family
named below and not across families.

**P1 (primary).** Pooled over the three indices on the 604 common dates, the hybrid has a lower mean daily QLIKE than
mcsGARCH, at 15 minutes (`P1_next15`) and at 30 minutes (`P1_next30`). Test: one-sided DK t test of a' mu > 0 (Section
4). Holm over the two. **Verdict:** both Holm p at most 0.05: "pooled gain confirmed"; exactly one: "partly confirmed"
(naming the horizon); neither: "not confirmed". (By Section 2 the verdict cannot be anything but "confirmed"; the
numbers that matter are the pooled t and its place between the bounds.)

**Secondary** (each its own family and verdict; none can change P1):

* **S1a, joint bootstrap.** The P1 hypotheses tested with a stationary bootstrap (Politis and Romano 1994) over dates
  that resamples **all three indices together** (the same dates for every index in every draw): mean block length
  **10 trading days**, **9,999 draws**, generator `numpy.random.default_rng([107, 10])`. p = (1 + number of draws in which
  the recentred pooled mean, m* - m, is at least m) / 10,000; the smallest possible p is 0.0001. Holm over the two
  horizons. Verdict: both Holm p at most 0.05: "bootstrap agrees". Robustness R3: mean blocks 5 and 20 (generators
  `[107, 5]` and `[107, 20]`).
* **S1b, homogeneity.** Null: the three mean differences mu_1, mu_2, mu_3 are equal. Wald statistic W = (R mu)'(R V R')^-1
  (R mu) with R the two successive differences and V the DK covariance of Section 4, multiplied by (T - 1)/T (the square
  of the HLN factor); F = W/2 against F(2, T - 1). Non-directional. Holm over the two horizons. Verdict: a Holm p at most
  0.05: "the gains differ" at that horizon; otherwise "no evidence that the gains differ". Not rejecting is not proof of
  equality; the write-up reports the three means and their difference.
* **S1c, pooled last hour.** As P1 for the 14:30 to 15:30 window, one test, p at most 0.05: "supported". (Certain by
  Section 2; reported for completeness.)
* **S1d, pooled expiry days.** As P1 on each index's own expiry days (Section 4), at 15 and 30 minutes, Holm over the
  two. Verdict: both Holm p at most 0.05: "supported"; one: "partly supported"; none: "not supported".
* **S1e, pooled edge over `mcsgarch_iv`.** As P1 with `mcsgarch_iv` as the benchmark, at 15 and 30 minutes, Holm over
  the two, same verdict wording as S1d.

**Robustness** (no verdict of their own; reported next to the registered result): R1 scaled differentials (P1, two
horizons); R2 bandwidth 10 (P1, two horizons); R3 bootstrap with mean blocks 5 and 20 (P1, two horizons each); R4
expiry days with equal weight per index-day (S1d, two horizons).

**Exploratory** (labelled exploratory, no verdicts): E1 pooled over Bank Nifty and Fin Nifty only (the two indices the
hybrid was not designed on), 15 and 30 minutes; E2 each index alone against `mcsgarch_iv` (project DM test), three
horizons, nine tests; E3 pooled last hour against `mcsgarch_iv`; E4 pooled last hour on expiry days against mcsGARCH;
E5 homogeneity in the last hour.

**Placebo (descriptive, no p-value).** Bank Nifty's differential series is shifted by 21 trading days and Fin Nifty's
by 42 (circularly), which breaks the shared dates but keeps each series' own behaviour. The correlations should then be
near 0 and the effective number near 3. If they are not, the effective-number machinery is picking up something other
than shared days.

## 6. Correlation and the effective number of indices (descriptive, the novel part)

For each target (15 minutes, 30 minutes, last hour) and benchmark (mcsGARCH, `mcsgarch_iv`), on the balanced 604-date
panel of daily differentials d_it:

* the **Pearson correlation matrix** (3 x 3), with **Spearman** as a robustness check against a few extreme days, the
  **long-run correlation** matrix (from the Newey-West covariance Omega with L = 5, the dependence the pooled test
  actually uses), and, for context, the correlation of the three indices' daily mcsGARCH QLIKE **levels**;
* the eigenvalues of the Pearson matrix;
* **effective numbers of indices.** The **registered headline is the design effect of the pooled test**:
  N_eff = 3 tr(Omega) / (1' Omega 1) = 3 x (variance of the pooled mean if the indices were independent) / (its actual DK
  variance). It is the number of independent indices that the pooled test is worth; with equally noisy indices it equals
  3 / (1 + 2 x average long-run correlation). It can exceed 3 if correlations are negative. Alongside it: Kish's
  N^2 / (1'R1) on the Pearson and on the long-run correlation matrix, the eigenvalue participation ratio (sum lambda)^2 /
  sum lambda^2 (Bretherton et al. 1999), Nyholt's (2004) 1 + (N - 1)(1 - Var(lambda)/N) and Li and Ji's (2005) sum
  f(|lambda|); and a bootstrap version of the design effect (bootstrap variances from S1a's draws).
* **"Nifty equivalents":** Omega_Nifty / (a' Omega a), how many times more precise the pooled mean is than Nifty's mean
  alone (above 1 means pooling helped), and the same for the best (GLS) weights if the three gains were equal.
* **What each index adds:** the same measures for each pair of indices, and N_eff(all three) minus N_eff(the pair
  without that index).
* **Uncertainty:** 95% percentile intervals from the S1a bootstrap draws (block 10, 9,999 draws), recomputing the
  correlation matrix and Omega on every resampled panel, for the Pearson and long-run correlations and all three-index
  measures, at 15 minutes, 30 minutes and the last hour against mcsGARCH. Block joins cut the serial dependence a little,
  so the long-run intervals are approximate.
* **SE comparison:** for P1, the DK standard error next to the standard error if the indices were independent and the
  naive one that treats all 1,812 index-days as independent.

Expiry days are not used for correlations (the three day sets differ).

## 7. What will be reported, whatever the result

In `docs/P107_POOLED_THREE_INDEX.md`, labelled REAL DATA, verdict first in plain words for a beginner (with plain
explanations of Driscoll-Kraay and of the effective number), then: the P1 table (per-index means, pooled change in
percent, pooled DK t, raw and Holm p) next to the Section 2 bounds; S1a to S1e with verdicts; robustness; exploratory;
the correlation matrices and effective numbers with intervals; the placebo; "Deviations from the plan"; caveats; and
what the study changes in the project's claims, including the sentence that the pooled test is a summary of evidence
already seen, not new evidence.

## 8. Outputs

`results/p107_pooled_test/`: `tests.csv` (every p-value), `verdicts.csv`, `pooled_estimates.csv`, `bootstrap.csv`,
`se_comparison.csv`, `correlations.csv`, `effective_n.csv`, `per_index.csv` (each index alone, statistics only),
`check_against_p104.csv` (our per-index numbers against P104's committed same-dates table; must match),
`daily_losses.csv` (the daily mean QLIKE of the three models for each index and date), `run_info.json` (plan commit,
seeds, input hashes, versions, run time) and `stdout.txt`.

## 9. Deviations policy

After this commit the design does not change. Any change is a deviation listed in the write-up with its reason and
whether real results had been seen. If `check_against_p104.csv` shows a mismatch, the run stops and the cause is found
and reported before any pooled number is written up. A bug found after the run is fixed with the smallest change, the
run is repeated, and both versions are reported. Anything added after seeing the results is labelled post hoc.

## 10. Known weaknesses

1. **Not new evidence** (Candour section). P1 and S1c are fixed in advance by the committed per-index statistics.
2. **The Nifty part is in-sample for the design** (Prior knowledge); E1 shows the two out-of-sample indices alone.
3. **The three models were fitted separately per index.** The pooled test asks whether the hybrid *method* wins on
   average across the three indices, not whether one model wins on a portfolio.
4. **604 common dates only.** Nifty's 188 later days and Bank Nifty's 56 later days are left out so that the panel is
   balanced.
5. **Long-run correlations from 604 days are noisy**, and the effective number is a ratio of noisy variances; the
   intervals show how noisy.
6. **Equal weights are a choice.** An index that is noisier (higher variance of d) counts as much as a quieter one;
   the GLS "Nifty equivalents" show what optimal weights would give, under the extra assumption of equal gains.

## 11. Code and tests

* `experiments/p107_pooled_test.py` (new): `count` mode (phase 1: calendar counts, hashes, bounds from the committed
  table) and `run` mode (phase 2). Pure functions: `dk_lag`, `kernel_sum`, `driscoll_kraay_cov`, `panel_ols_dk`,
  `index_means_dk`, `combo_test`, `wald_equal`, `long_run_cov`, `effective_numbers`, `stationary_bootstrap_indices`,
  `bootstrap_p_greater`, `boot_index_means`, `boot_dependence`. It uses `daily_losses`, `diebold_mariano` and `holm`
  from `niftygex.evaluation` unchanged. Nothing in `src/` is changed.
* `tests/test_p107_pooled_test.py` (new), SYNTHETIC data only, 21 tests: bandwidth rule; a hand-calculated 3-date,
  2-index example; the identity with Newey-West on the cross-sectional mean; the identity with the project's DM test for
  one index; agreement with statsmodels `hac-groupsum` (balanced regression and unbalanced index dummies); size under
  strong cross-sectional dependence (DK near 5%, naive test above 15%) and power; Wald test (equals the squared t for two
  indices, size and power); effective numbers at their extremes, under equicorrelation and as the variance ratio of the
  pooled test; the bootstrap (block lengths, p formula, joint resampling, dependence measures); the Section 2 bounds
  (formula, and the floor "pooled t at least the smallest per-index t" on synthetic panels); and an end-to-end run on
  synthetic forecast files with different date sets.
* **Checks run on 3 November 2026:** `tests/test_p107_pooled_test.py`: 21 passed. The full suite is run before the
  phase-2 push.
