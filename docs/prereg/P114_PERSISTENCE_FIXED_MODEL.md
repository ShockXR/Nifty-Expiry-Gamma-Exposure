# P114 pre-registration: does the fall in intraday persistence survive when mcsGARCH is fixed?

**Status.** Written on 3 November 2026 (phase 1) by an AI agent (Claude) working for Shreyan Thanki. **This design
is committed to git before any outcome of this study has been computed.** No extended model (with any of P109's
fixes) has been estimated on any before or after window, or on any rolling window, and no bootstrap draw, test
statistic or drift measure of this study has been computed on real data. The only real-data computations so far
are the reproduction checks of Section 12: the replication's own mcsGARCH (the reference) was refitted through this
study's code and compared with the committed Table 2 numbers, and P109's forecasts were recomputed for two refit
blocks and compared with P109's own run. Neither involves a new model on these windows. The script and its tests
were also run end to end on a simulated market (SYNTHETIC). The commit that adds this file is the time stamp; phase
2 (the run) starts only after that commit exists and has been pushed. The code that phase 2 runs
(`experiments/p114_persistence_fixed_model.py`, new code at the end of `src/niftygex/volatility.py`) and its tests
(`tests/test_p114_persistence_fixed_model.py`, synthetic data only) are in the same commit. Phase 2 writes
`results/p114_persistence_fixed_model/` and `docs/P114_PERSISTENCE_FIXED_MODEL.md`. All results will be REAL DATA.

## In plain words

The anchor paper (Chhimwal, Pandey and Pandey 2025) fits a volatility model called mcsGARCH to one-minute Nifty
and Bank Nifty returns, before and after the expiry-day change. One of its findings that our replication confirms
is that **intraday volatility shocks die out much faster after the change**. The model's "persistence" alpha +
beta gives the half-life of a shock: on our inferred before window it is 359 minutes for Nifty, against 72 minutes
after (`docs/REPLICATION_NSE.md`, finding 4).

Study P109 found that mcsGARCH is misspecified in ways that matter for forecasting. Its minute-by-minute part has
no slowly moving level inside the day, it carries yesterday's last minutes straight into today's open, and its
daily level is stale. Such a model can show a persistence close to 1 for a reason that has nothing to do with how
long single shocks last: if the day's volatility level drifts and the model has no other way to follow it, its
minute-by-minute part must stay high (or low) for hours, and that looks like long-lived shocks. If the before window
had more of this drift than the after window, part or all of the paper's "faster decay" could be an artefact.

**This study asks: does the fall in intraday persistence survive when the intraday model includes P109's fixes?**
We refit the replication's model exactly as in Table 2 (the **reference**) and the same model with P109's fixes
(the **fixed model**) on the same windows, measure how long one shock lasts in each, and test the change with the
replication's own bootstrap.

## Prior knowledge (what was known when this was written)

All REAL DATA, from committed files.

* **The replication's Table 2** (`results/replication_nse/table2_mcsgarch.csv`, `table2_persistence.csv`,
  `table2_persistence_change.csv`). ARMA(1,1)-mcsGARCH(1,1) with Student-t errors, daily level from an
  AR(1)-EGARCH(1,1)-t, 360 minutes a day (09:15 to 15:14) with the overnight return in the first minute.
  * Nifty: inferred before window (14 November 2022 to 10 February 2023, 64 days) alpha 0.0246, beta 0.9735,
    alpha + beta 0.99807, half-life 358.6 minutes (bootstrap 95% interval 103 to 701); after window (1 March to 31
    May 2024, 59 days) alpha 0.0498, beta 0.9407, alpha + beta 0.99048, half-life 72.4 (37 to 139). Change in alpha
    + beta, after minus before: -0.0076, bootstrap 95% interval -0.0163 to -0.0006, 99% of draws show a fall;
    likelihood-ratio statistic 16.23, p 0.00006.
  * Bank Nifty: half-life 162.0 before (51 to 302) and 38.0 after (11 to 102); change in alpha + beta -0.0138,
    interval -0.0591 to +0.0023 (94.25% of draws show a fall); likelihood ratio 18.27, p 0.00002.
  * Stated before window (1 November 2023 to 29 February 2024, 82 days): Nifty half-life 383.3, change interval
    -0.0164 to +0.0011; Bank Nifty 68.9, interval -0.0535 to +0.0150.
  * Moving the 64-day window by a few days changes the Nifty before half-life from 318 to 370 minutes
    (`table2_window_sensitivity.csv`). P108 (`docs/P108_BLIND_WINDOW.md`) prefers 22 November 2022 to 20 February
    2023 as the before window; the two candidates share 58 of 64 days.
* **P109** (`results/p109_mcsgarch_extensions/params_summary.csv`, `docs/P109_MCSGARCH_EXTENSIONS.md`), walk-forward
  refits on Nifty 2023 to 2026 (means over 38 refits): mcsGARCH alpha 0.045 and beta 0.950. X3 alone (today's
  running mean): alpha 0.056, beta 0.914, gam_x 0.024. The combination: alpha 0.035, beta 0.923, gam_x 0.018,
  gam_l 0.033, delta 0.09 (only 9% of the intraday part's deviation from its average survives the night). X1, X2
  and X3 improve forecasts; X4 (leverage) does not (alone -0.27 and -0.39 points; removing it from the combination
  changes the loss by +0.15 and +0.16 points, not significant). So in P109's long samples, the fixes already lower
  beta and cut the overnight carry. Nothing is known about the before and after windows themselves under the fixes.

## 1. The models

All models use the replication's data and steps unchanged: the two-index minute grid of
`replicate_paper.build_grids` (days on which both indices have at least 300 bars in the session), 360 minutes from
09:15 to 15:14, one-minute log returns with the overnight return in the first minute, the window's very first
return dropped, the ARMA(1,1) mean by conditional least squares (fitted first), Student-t errors, the diurnal
curve as the plain average of eps^2 / h by minute (no smoothing, no separate expiry-day curve), and everything
estimated **in sample on each window**, as the replication does for Table 2.

| Name | Daily level h_t | Intraday recursion | Role |
|---|---|---|---|
| `ref` | AR(1)-EGARCH(1,1)-t on daily returns since April 2017, fitted up to the window's last day | GARCH(1,1): q_{n+1} = omega + alpha x_n^2 + beta q_n, run straight through the night | reference: exactly the replication's Table 2 fit (`replicate_paper.fit_period`) |
| **`F`** | **X1**: log-HAR of daily realized variance (`volatility.har_daily_variance`), fitted on days since April 2017 up to the window's last day | **X3** GARCH-X term gam_x z_n (z = today's running mean of x^2) and **X2** overnight decay q_open = qbar + delta (carry - qbar), delta estimated (`ext_q2`, `fit_garch_ext` with P109's two registered starting points) | **the fixed model (primary)** |
| `F4` | X1 | X2 + X3 + **X4** leverage term gam_l x_n^2 1{eps_n < 0} | secondary: the likelihood part of P109's combination |
| `X1`, `X2`, `X3` | one fix at a time (X2 and X3 with the EGARCH level) | as above, one term each | exploratory, point estimates only |

* **Why X4 is not in the primary model.** P109 found that X4 does not improve forecasts, and it has nothing to do
  with the drift mechanism this study is about. It also makes the response to a shock depend on the shock's sign.
  It is kept as a secondary model (F4), with the sign averaged (Section 2).
* **Why X5 is left out.** X5 multiplies the model's forecast at each forecast time of day by a factor learnt from
  out-of-window forecast errors. It does not enter the likelihood or the recursion, so it changes no parameter,
  no alpha + beta and no impulse response. It has no role in an in-sample persistence study.
* **Daily realized variance for X1**: the sum of squared one-minute log returns in the 360-minute session, with the
  day's first bar measured from its own open, so the overnight gap is left out (as in P109, whose session ran to
  15:29). `daily_rv` in the script.

## 2. The persistence measures

Two measures are computed for every model, window and index.

1. **alpha + beta** (comparable with the paper and Table 2). In F it is the weight of the last minute's variance and
   squared return only; it leaves out the running-mean term and the overnight decay. Also reported, not tested: the
   stationarity sum alpha + beta + gam_x (+ 0.5 gam_l).
2. **The impulse-response half-life** (`volatility.ext_impulse_response`, `irf_half_life`,
   `ext_persistence_summary`). At **12:15 (minute 180 of 360)** the squared normalised residual x^2 is set one unit
   above its conditional expectation; every later x^2 is replaced by its expectation. Because the model is linear in
   (q, x^2), the response of the expected variance is computed **exactly by the recursion** (no simulation) and does
   not depend on the shock's size or the current level. The model's own clock is used: one step per minute within a
   day, the night is one step (q at the next open = qbar + delta (carry - qbar)), and today's running mean restarts
   each day. With leverage (F4) the shock's sign is averaged (0.5 gam_l). Let r_1 be the response one minute after
   the shock. The **half-life** is the number of minutes after that at which the response first falls to half of
   r_1, interpolated log-linearly between minutes; if it never does within 100,000 trading minutes it is infinite. Its
   **equivalent persistence** is rho = 0.5 ** (1 / half-life), the alpha + beta of a GARCH(1,1) with the same
   half-life. For the reference (and any GARCH(1,1)) the half-life is exactly ln(0.5) / ln(alpha + beta), the
   paper's half-life, and rho = alpha + beta (tested).

**The impulse-response half-life is primary.** The paper's claim is about how fast shocks die out. In the fixed
model one shock raises today's running mean, which keeps raising the variance for the rest of the day, and the
overnight decay cuts whatever is left at the close. Only the impulse response counts both. alpha + beta alone
would also make the before-after comparison depend on how the model splits persistence between beta and gam_x. And
for the reference model the primary measure is exactly the paper's half-life, so the two models are compared on one
scale.

Also reported for each fit (descriptive): the half-life for shocks at 10:15 and 14:15; the share of the first
response left at the close of the shock's day and at the next open. Note: if delta is small, a shock at 12:15
cannot have a half-life much beyond the close (about 179 minutes later), because most of what is left is removed
overnight. This is part of the fixed model, not a defect of the measure, and the share left at the close shows it.

## 3. Windows and indices

| Window | Days | Role |
|---|---|---|
| inferred before: 14 November 2022 to 10 February 2023 | 64 | **primary** before window |
| stated before: 1 November 2023 to 29 February 2024 | 82 | secondary |
| P108's blind window: 22 November 2022 to 20 February 2023 | 64 | robustness |
| after: 1 March to 31 May 2024 | 59 | the after window of every comparison |

**Nifty is primary, Bank Nifty secondary.**

## 4. The test: the replication's day-block bootstrap (primary) and the likelihood-ratio test (secondary)

**Bootstrap** (`boot` stage). Exactly the replication's (`replicate_paper._boot_persistence_worker`): for each
window and index, 400 resamples of circular 5-day blocks of days, drawn with the replication's generator and seeds
(seed 1000 i + c for chunk c = 0..3 of 100 draws, i the window's position in the replication's list: stated Nifty
0, stated Bank Nifty 1, inferred Nifty 2, inferred Bank Nifty 3, after Nifty 4, after Bank Nifty 5; P108's window
Nifty 6, Bank Nifty 7). On each resample the diurnal pattern is re-estimated from the full-sample ARMA residuals and
the model's daily level, the normalised residuals are rebuilt (today's running mean within each resampled day),
and the intraday model is refitted: `fit_garch11` for the reference (this reproduces the replication's draws bit
for bit, tested on simulated data and checked on the real draws in phase 2), `fit_garch_ext` for F and F4 (P109's
two starting points). **Every model is refitted on the same resampled days in each draw**, so the reference and the
fixed model are paired. Before and after draws are independent and paired by draw number, as in the replication.
Bootstrapped: `ref` and `F` on all eight window-index keys; `F4` on the inferred and after windows.

* **Test statistic**: D = rho_after - rho_before, the change in equivalent persistence (negative = shocks die out
  faster after, the paper's direction), with the **95% percentile interval** of its draws.
* **Bootstrap p** (for `tests.csv`): one-sided for "persistence fell", p = (1 + draws with D* >= 0) / (1 + draws).
  An interval entirely below zero corresponds to p below about 0.025.
* Draws in which a measure cannot be computed (first response 0) are dropped from that interval and counted.

**Likelihood-ratio test** (`lr` stage, secondary). As the replication's `_joint_lr_worker`: the two windows are
fitted jointly with one common value of the persistence measure, each keeping its own other parameters; the common
value is searched on the log half-life scale over [log of the smaller half-life - 0.5, log of the larger + 0.5]
(bounded scalar search, tolerance 0.001). The constrained fit is `volatility.fit_garch_ext_fixed`: for alpha + beta,
alpha + beta is held fixed (free: the alpha share, gam_x, delta, nu and the unconditional level); for the
impulse-response half-life, beta is solved from the half-life given the other parameters (the half-life rises with
beta). L-BFGS-B, then a Nelder-Mead polish, one start at the window's own fit. LR = 2 (L_before + L_after - L_joint),
where L is the larger of the unconstrained fit and the best constrained value found for that window; p from
chi-square(1), two-sided, as in the replication. With every fix switched off this code reproduces the replication's
committed LR statistics to four decimals (Section 12). The LR test trusts the minute-by-minute likelihood, which
ignores the dependence that the bootstrap respects, so the bootstrap is primary.

## 5. Hypotheses, tests and decision rule

**Primary hypothesis (P1, 1 test).** Under the fixed model F, Nifty's impulse-response half-life is shorter after
than on the inferred before window (D < 0). Test: the bootstrap. `P1_nifty_inferred_F_irf_boot`.

**Decision rule** (fixed now; applied by `verdict_rule` in the script), on Nifty, inferred window, model F:

| Outcome | Rule |
|---|---|
| **survives** | the 95% bootstrap interval for D lies entirely below 0 |
| **weakened** | D < 0, but the interval reaches 0 or above |
| **gone** | D >= 0 and the interval reaches below 0 |
| **reversed** | the interval lies entirely above 0 |

**How much of the original fall the fixed model keeps** (reported with every outcome): the kept share
K = log(HL_F,before / HL_F,after) / log(HL_ref,before / HL_ref,after), the fall in log half-life under the fixed
model as a share of the fall under the reference (the reference's is log(358.6 / 72.4) for Nifty). Its 95%
interval comes from the paired draws (both models on the same resampled days); also reported is the interval of
the difference of the two log falls, which is stable when the reference's fall in a draw is near zero. A size label
goes with the outcome: "keeps most of the fall" if K >= 0.5, "keeps less than half" if 0 < K < 0.5, "keeps none"
if K <= 0 (point estimate). Infinite half-lives are capped at 100,000 minutes for logs and ratios.

**What would support the paper's finding:** "survives" (the faster decay after the change is present even when the
intraday model has a running level, an overnight reset and a better daily level), more strongly with "keeps most of
the fall". "Survives" with "keeps less than half" means the direction is real but most of the size of the printed
change (a fivefold fall in half-life) is an artefact of the misspecification.

**What would contradict it:** "gone" or "reversed": once the model can follow the day's level, shocks do not die
out faster after the change (in this window, with this model). "Reversed" would be the strongest contradiction.

**Neither:** "weakened": the fixed model points the same way but cannot confirm the fall with 64 + 59 days. We
would then say the paper's finding rests on the misspecified model and is not robust to the fixes, without
calling it false.

**What we expect** (stated in advance; it changes no rule): from P109's walk-forward parameters, both windows will
have much shorter half-lives under F than under the reference (tens of minutes rather than hundreds), and the fall,
if any, will be smaller than the reference's in log terms. Our guess is "weakened", with a kept share between 0 and
0.5. This is a guess, not a rule.

**Secondary families** (Holm within each; they cannot change the primary verdict):

* **S1 (3 tests), Nifty, inferred window, model F, other test or measure**: LR test of equal impulse-response
  half-life (`S1_nifty_inferred_F_irf_lr`); bootstrap of the change in alpha + beta (`S1_nifty_inferred_F_ab_boot`);
  LR test of equal alpha + beta (`S1_nifty_inferred_F_ab_lr`).
* **S2 (4 tests), Bank Nifty, inferred window, model F**: bootstrap and LR, for the impulse-response half-life and for
  alpha + beta.
* **S3 (4 tests), stated before window, model F, impulse-response half-life**: bootstrap and LR, Nifty and Bank Nifty.
* **S4 (2 tests), model F4, inferred window, impulse-response half-life**: bootstrap, Nifty and Bank Nifty.

The same decision rule labels every secondary comparison (`verdict.csv`), as a description.

**Robustness R1 (2 tests, Holm)**: P108's blind window against the after window, model F, impulse-response
half-life, bootstrap, Nifty and Bank Nifty.

**Reference rows (not new tests).** The reference model's bootstrap and LR results on the replication's windows
repeat committed numbers; they are reported for comparison and as reproduction checks, not as new p-values in
`tests.csv`. The reference on P108's window is new but descriptive (no test).

**Exploratory (no adjustment).**

* **E1, rolling windows.** Windows of 64 trading days, the first starting on 14 November 2022 (the inferred window),
  then every 21 trading days, up to the last full window of each index's data (Nifty to 2 November 2026, Bank Nifty
  to 22 April 2026). Because the Bank Nifty minute data end in April 2026, each index uses its own one-index grid
  (the replication's grid rules for one index: days with at least 300 bars; `index_grid`); the daily EGARCH and HAR
  levels are fitted from April 2017 (Nifty) or January 2017 (Bank Nifty) up to each window's last day. Models `ref`
  and `F`, point estimates only.
* **The drift measure (stated in advance).** For a window, let m_t be the mean over the day's valid minutes of the
  squared normalised residual x^2 = eps^2 / (h_t s_i) of the **reference** fit: the day's level relative to what the
  reference's daily level and diurnal curve expect. **Drift** = the standard deviation over the window's days of log
  m_t. A window in which the day's level wanders more has a larger drift. Also reported: the lag-1 autocorrelation of
  log m_t, and both measures for F's own residuals, for every window including the before and after windows.
* **E1 tests (4)**: for each index, the Spearman rank correlation across windows between the drift and (a) the
  reference's log half-life, (b) the fixed model's log impulse-response half-life. p-values only on the
  non-overlapping windows (every fourth window, 84 days apart); the correlation over all windows is descriptive. If
  the artefact story is right we expect (a) to be clearly positive and (b) to be weaker.

Every p-value goes in `results/p114_persistence_fixed_model/tests.csv` with the columns `test_id, hypothesis, role,
sample, estimate, statistic, p_raw, sidedness, p_method, p_adjusted, adjust_method` (as P109). Any test added after
this commit is labelled "post hoc".

## 6. The exact commands

From the repository root, at the commit that adds this file, with the raw data listed in `data/MANIFEST.csv` in
`data/raw`. Work files go to `data/processed/p114/` (git-ignored); the bootstrap writes one file per window and
index and does not redo a finished one.

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py inputs
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py fit
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py lr
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py rolling
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py boot --workers 1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_persistence_fixed_model.py analyze --plan-commit <this commit>
```

The bootstrap runs the keys in this order: inferred Nifty, after Nifty, inferred Bank Nifty, after Bank Nifty
(these four with F4), stated Nifty, stated Bank Nifty, P108 Nifty, P108 Bank Nifty. One worker process; a second
worker is added only if the projected finish is after 16:00 UTC (the seeds are fixed per key, so this changes no
number). `analyze` writes `tests.csv`, `verdict.csv`, `persistence.csv` (every model, window and index: parameters,
both measures, bootstrap intervals), `persistence_change.csv` (every comparison: changes, intervals, bootstrap p,
fall factors, kept shares, LR results), `fits.csv`, `bootstrap_draws.csv`, `lr.csv`, `irf_curves.csv`,
`rolling.csv`, `rolling_correlations.csv`, `reproduction_table2.csv`, `reproduction_bootstrap_lr.csv`,
`fig_irf.png`, `fig_rolling.png` and `run_info.json` (plan commit, settings, input hashes, run times, versions).

## 7. Checks inside the run

* `inputs` refuses to continue unless the minute-bar files match `data/MANIFEST.csv`.
* The reference fit must equal the committed Table 2 values (`reproduction_table2.csv`); the reference bootstrap
  draws must give the committed percentiles of alpha + beta and of its change (`reproduction_bootstrap_lr.csv`);
  the reference LR statistics through this code must match the committed ones. A failure is reported as a
  deviation.
* The ARMA residuals of every model are identical (checked in `fit`).

## 8. What will be reported, whatever the result

In `docs/P114_PERSISTENCE_FIXED_MODEL.md`, for a beginner reader, every number labelled REAL DATA (or SYNTHETIC) and
taken from a file in `results/p114_persistence_fixed_model/`: the plain-language answer first; the primary verdict
with its size label; a table of persistence and half-life before and after under both models, with intervals; the
fitted parameters in plain words; the secondary, robustness and exploratory rows with their labels; the rolling
windows and the drift; what it means for the paper's claim; deviations; caveats; files and how to reproduce.

## 9. Deviations policy

* After this file is committed the design does not change. Any change is a deviation, listed with its reason and
  with whether any outcome had been seen when it was decided. No specification search: no other shock minute,
  window, model, starting point or measure is promoted after outcomes are seen.
* A crash is fixed with the smallest change that lets the run finish, and only if it changes no number already
  computed.
* A fit that does not converge is reported (`converged`) and used as it is.
* If the bootstrap cannot finish by 16:30 UTC on 3 November 2026, the unfinished keys are reported as "not
  computed" (the order above puts the primary keys first), and the rows that need them are left out of
  `tests.csv` and named in the write-up.

## 10. Known weaknesses

1. **Small windows.** 64 and 59 days. Half-lives near the before window's are poorly determined (the reference's
   interval runs from 103 to 701 minutes), and short windows also limit how well gam_x and delta can be estimated.
2. **In-sample, like the paper.** The fixed model's parameters come from the same window they describe. This is the
   paper's design and the replication's; no forecasts are evaluated here.
3. **The fixes were chosen for forecasting** (P109), on 2023 to 2026 data, not for this question. They are the
   natural candidates because they target exactly the drift mechanism, but other ways to model a moving level (for
   example a two-component intraday GARCH or a realized-GARCH) are not tried.
4. **The overnight first minute.** The replication's first-minute return carries the overnight gap, while P109's
   did not. The fixed model's running mean therefore starts each day with that minute's normalised square.
5. **The half-life of a mid-day shock is capped by the night** when delta is small (Section 2). The share left at
   the close and the half-lives for shocks at 10:15 and 14:15 are reported so that this is visible.
6. **The before window is uncertain** (Section 3); P108's window is the robustness row.
7. **The bootstrap keeps the full-sample ARMA residuals and daily level fixed** and joins resampled days at block
   boundaries, as the replication does.
8. **The LR test's profile** uses one start and a bounded search; it reproduces the replication when the fixes are
   off, but could miss a better constrained maximum with them on. It is secondary for this reason too.
9. **Rolling windows use one-index grids**, which differ slightly from the replication's two-index grid in the
   history used for the daily levels (Section 12), so the first rolling window need not equal the Table 2 fit.

## 11. Code

* `src/niftygex/volatility.py`, new code after all existing code, nothing above changed:
  `ext_impulse_response`, `irf_half_life`, `equivalent_persistence`, `ext_persistence_summary`,
  `solve_beta_for_half_life`, `fit_garch_ext_fixed` (and the constants `IRF_MAX_STEPS`, `PERSISTENCE_CAP`).
* `experiments/p114_persistence_fixed_model.py`, new: stages `inputs`, `fit`, `boot`, `lr`, `rolling`, `analyze`;
  `--synthetic` runs all of them on a simulated two-index market (labelled SYNTHETIC). It imports the
  replication's script for the grid, the windows, the daily EGARCH and the reference fit.
* `experiments/p114_plan_checks.py`, new: the checks of Section 12.
* `tests/test_p114_persistence_fixed_model.py`, new, synthetic data only (18 tests): the GARCH(1,1) impulse
  response and its half-life ln(0.5) / ln(alpha + beta); the extended impulse response against a deterministic loop
  and against a Monte Carlo of the full model (night step and running-mean restart included); the half-life rules;
  the beta solver; the profile likelihood against the replication's and against the unconstrained fit; the joint LR
  against the replication's worker; the bootstrap reproduces the replication's draws bit for bit; the daily realized
  variance leaves out the overnight gap; the drift measure; the decision rule and the bootstrap p; an end-to-end
  dry run of every stage.

## 12. Checks run before this file was committed (3 November 2026)

Output of `PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_plan_checks.py --p109-rerun <scratch folder>
--p109-committed <P109's work folder>` is committed as
`results/p114_persistence_fixed_model/plan_checks_stdout.txt`. All REAL DATA, reference model only.

* **Table 2 reproduces exactly** through this study's driver: 24 of 24 values (omega, alpha, beta and nu, on the
  three windows and two indices) are identical to the committed file.
* **The replication's LR statistics reproduce** with this study's profile code and every fix switched off:
  inferred Nifty 16.2299, inferred Bank Nifty 18.2722, stated Nifty 14.3819, stated Bank Nifty 3.6290, equal to the
  committed values to four decimals, with the same common half-lives.
* **Grids for the rolling windows.** Up to 30 June 2024, the Nifty one-index grid has one day more than the
  two-index grid (24 February 2021) and the Bank Nifty one-index grid 62 days more (January to March 2017, before the
  Nifty minute data start); no day of the two-index grid is missing from either.
* **P109's forecasts are unchanged** by this study's new code: P109's driver was rerun for the first and the last
  Nifty refit block (blocks starting at days 150 and 927, all 12 variants) and compared, as text, with the block
  files of P109's own run: 18,900 of 18,900 and 13,500 of 13,500 cells identical, and both parameter files
  identical.
* **Dry run** (SYNTHETIC: two simulated indices, 420 days, a drifting daily level, the intraday beta falling at day
  330): all six stages ran. On these simulated data the reference's half-life was several times the fixed model's,
  which is the artefact this study looks for.
* **Tests.** `PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_p114_persistence_fixed_model.py`: 18
  passed.
