# P109 pre-registration: what does the hybrid's correction fix in mcsGARCH?

**Status.** Written on 3 November 2026 (phase 1) by an AI agent (Claude) working for Shreyan Thanki. **This design
is committed to git before any outcome of this study has been computed.** No forecast of any extended model has
been made on real data, and no loss, gain, share, test statistic or bootstrap draw of this study has been computed
on real data. The only real-data computation so far is the reproduction check in Section 11: this study's driver
refitted the committed mcsGARCH on Nifty, Bank Nifty and Fin Nifty and its forecasts and targets were compared
with the committed ones, number by number, without computing any loss (those forecasts and their losses are
public in `results/pipeline_nse_v2/` and the two other committed runs). The script and its tests were also run end
to end on a simulated market (SYNTHETIC). The commit that adds this file is the time stamp; phase 2 (the run)
starts only after that commit exists and has been pushed. The code that phase 2 runs
(`experiments/p109_mcsgarch_extensions.py`, one new code path in `src/niftygex/volatility.py`) and its tests
(`tests/test_p109_mcsgarch_extensions.py`, synthetic data only) are in the same commit. Phase 2 writes
`results/p109_mcsgarch_extensions/` and `docs/P109_MCSGARCH_EXTENSIONS.md`. All results will be REAL DATA.

## In plain words

Our hybrid forecasts how much the Nifty will move over the next 15 or 30 minutes. It takes the anchor paper's
model (mcsGARCH) and multiplies its forecast by a correction learnt by LightGBM, a machine-learning model. Its
loss (QLIKE) is 5.80% and 7.65% lower than mcsGARCH's. Study P106 showed that the correction gets all of this
gain from mcsGARCH's own kind of information (past one-minute returns, the time of day and the expiry-day flag):
a hybrid that sees only that information is 6.06% and 8.01% better than mcsGARCH. So the learnt correction is
fixing something that mcsGARCH itself gets wrong.

This study asks **what that something is**. We write down, in advance, five small changes to mcsGARCH. Each one
fixes one specific thing we suspect mcsGARCH gets wrong, and each has a reason from earlier results:

1. **X1, the daily level.** mcsGARCH takes each day's volatility level from a model of daily returns (an EGARCH).
   We replace it by a forecast built from the realized variance of the previous day, week and month (a HAR
   model), which uses the minute data and is known to forecast daily volatility better.
2. **X2, the overnight break.** mcsGARCH's minute-by-minute component runs straight from yesterday's last minute
   into today's first minute, as if the 17-hour night were one minute. We let it fade toward its average over the
   night, by an amount the data choose.
3. **X3, the day's level so far.** The minute-by-minute component remembers only about the last 20 minutes. We
   add today's realized variance so far as an input, so the model can learn the day's level faster.
4. **X4, falls versus rises.** The minute-by-minute component treats a fall and a rise of the same size alike.
   We let falls raise volatility more (a "leverage" term).
5. **X5, the time of day.** We multiply mcsGARCH's forecast by one factor for each forecast time of the day,
   learnt from the training days, so that its average level is right at every time of day.

Then we fit all five together (the **combination**) and ask: **does the combination beat mcsGARCH, and how much
of the hybrid's gain does it recover?** If a simple, readable model recovers most of the gain, we can tell the
professor in plain words what the hybrid is correcting. If it recovers little, the learnt correction does
something these five changes do not, and we say so.

## Why this study

P106 (`docs/P106_GAIN_DECOMPOSITION.md`) attributed the hybrid's gain to groups of inputs. It did not say what
the correction does to mcsGARCH's forecast. A black-box correction that cannot be described is weak as a research
finding and hard to defend in front of a referee. An interpretable answer ("mcsGARCH's daily level is too slow",
"its forecast at the open carries yesterday's close") is far more useful, and it would let the paper report an
improved benchmark rather than only a machine-learning wrapper. A negative answer is also useful: it would show
that the hybrid's gain is not a simple misspecification that any careful econometrician would fix.

## Prior knowledge (what was known when this was written)

All REAL DATA, from committed files. Gains are in points of mcsGARCH's mean QLIKE (the hybrid's gain is
100 x (L_mcsGARCH - L_hybrid) / L_mcsGARCH).

* **The hybrid's gain** (`results/pipeline_nse_v2/scoreboard_*.csv`, 792 test days, 10 August 2023 to
  2 November 2026): 5.80 points at 15 minutes, 7.65 at 30 minutes and 6.36 in the last hour.
* **What carries it** (P106, `results/p106_gain_decomposition/contributions.csv`, `shares.csv`): removing the
  realized-history inputs costs 2.00 and 2.90 points (34% and 38% of the gain); removing the time-of-day inputs
  costs 1.34 and 1.92 points (23% and 25%); the option-implied, VIX and calendar groups add nothing; the hybrid
  that sees only mcsGARCH's information (`mcsinfo`) is 6.06 and 8.01 points better than mcsGARCH. The five group
  shares add up to only about half the gain (49% and 48%). LightGBM always fits an overall level (an intercept)
  for its correction, so part of the other half may be a plain level correction that no group removal can take
  away.
* **Where in the day** (`results/pipeline_nse_v2/gain_by_anchor_*.csv`, `docs/MODEL_ITERATIONS_REAL.md`
  Section 4.4): at the 09:15 forecast the hybrid is 13.3% and 23.9% better than mcsGARCH, at all later forecasts
  5.1% and 6.0%. The 09:15 anchor is 4% of forecasts but brings 20% and 29% of the hybrid's QLIKE reduction. Next
  largest: 09:30 (-14.1% and -17.2%), 09:45 (-6.5% and -10.4%) and 15:00 (-12.5% and -14.8%); between 10:15
  and 14:45 mostly -1% to -8%.
* **What the trees split on** (`feature_importance_<target>.csv`, `hybrid_base`, share of split gain): mcsGARCH's
  own forecast 12.3% and 11.6% (15 and 30 minutes), the signed direction of the last 15 minutes' returns
  (`ofi15`, a realized-group input) 8.4% and 4.9%, the time of day (`minute_frac`) 7.3% and 8.1%, mcsGARCH's
  diurnal share of the window 7.6% and 5.4%, realized variance of the last window 6.1% and 3.0%, today's realized
  variance so far (`log_rv_today_pm`) 5.9% and 5.7%, the same window yesterday 4.7% and 5.3%, last month's realized
  variance 3.2% and 4.3%. Split gain says what the trees use, not what they would lose without it.
* **mcsGARCH's intraday component** (`mcsgarch_params.csv`, 38 refits at 15 minutes): alpha 0.025 to 0.052, beta
  0.943 to 0.975, persistence 0.994 to 0.99999. So the intraday component's memory of past squared returns is
  short (beta 0.945 means a weight that halves about every 12 minutes), while its pull back to the average is
  slow. Its recursion runs from the last minute of one day straight into the first minute of the next
  (`src/niftygex/volatility.py`, as in the R package rugarch), so the 09:15 forecast carries the last half hour
  of the previous session.
* **Other daily levels tried before** (`results/referee_forecast/fair_benchmarks.csv`, all days, % of mcsGARCH's
  QLIKE): the option-implied daily level (`mcsgarch_iv`) -1.05% and -1.88%; a VIX-based level -0.57% and -1.20%;
  an EGARCH with log VIX (`mcsgarchx_vix`) +0.48% and +0.68%; the paper's EGARCH with a strictly real-time
  fallback rule (`mcsgarch_realtime`) +5.46% and +6.94% (the committed benchmark uses an EWMA daily level in two
  early refits, 42 test days, by a rule that looked ahead; `docs/REFEREE_RESPONSE_FORECAST.md`, side finding 1).
  The ridge log-HAR on the hybrid's inputs (`linear_base`) is -2.22% and -3.93%. None of these is a HAR daily
  level inside mcsGARCH.
* **Bank Nifty and Fin Nifty** (P91, P104, P106): the hybrid beats mcsGARCH by 3.92 and 5.03 points (Bank Nifty,
  660 test days) and 4.90 and 6.09 points (Fin Nifty, 604 test days); the realized and time-of-day groups lead
  there too.

No extended mcsGARCH of this study has been fitted on real data.

## 1. The five extensions and the combination

All five keep everything else of the committed mcsGARCH (`McsGarch(expiry_diurnal=True, smooth=5)` in
`run_walkforward`): zero mean, Student-t errors, the diurnal curve as the average of eps^2 / h by minute, with a
separate expiry-day curve, smoothed by a centred 5-minute moving average, re-estimated at every refit. All five use
only mcsGARCH's own information (past one-minute returns of the index, the time of day, the expiry-day flag) and
data before the forecast time.

| Name | Misspecification it targets | Change to mcsGARCH | New parameters | Reason from existing evidence |
|---|---|---|---|---|
| **X1** `x1_har` | the daily level is too stale and too noisy | h_t = exp(b0 + b_d log RV_{t-1} + b_w log RV^w_t + b_m log RV^m_t), a log-HAR (Corsi 2009) of daily realized variance (sum of squared one-minute returns, 09:15 to 15:30), RV^w and RV^m the means over the previous 5 and 22 days; OLS on the training days, replacing the daily AR(1)-EGARCH(1,1) | 4 (OLS) | the realized group (yesterday's, last week's and last month's realized variance) carries 34% and 38% of the gain; at 09:15, where mcsGARCH's forecast is essentially a fixed factor times the daily level, the hybrid gains most; the EGARCH reacts to one squared daily return and needed an EWMA fallback in two refits |
| **X2** `x2_overnight` | yesterday's close leaks into today's open | at the first minute of each day the intraday component is pulled toward its unconditional level: q_open = qbar + delta (q_carry - qbar), with delta in (0, 1) estimated by maximum likelihood (mcsGARCH: delta = 1) | 1 | the 09:15 and 09:30 forecasts carry the largest anchor gains; at 09:15 the realized inputs the hybrid sees (last 15, 60 minutes, last window) are yesterday's close; mcsGARCH's recursion treats the night as one minute |
| **X3** `x3_todayrv` | the intraday component forgets the day's level too fast | GARCH-X: q_{n+1} = omega + alpha x_n^2 + beta q_n + gamma_x z_n, with z_n today's running mean of the normalised squared returns x^2 up to minute n (at the day boundary, the whole previous day's mean) | 1 | today's realized variance so far and the last 60 minutes are among the hybrid's most used inputs; the intraday component's weight on past squared returns halves about every 12 minutes, while the hybrid gains 2% to 8% at most mid-day forecasts |
| **X4** `x4_leverage` | falls and rises move volatility alike | GJR term: q_{n+1} = ... + gamma_l x_n^2 1{eps_n < 0} | 1 | the signed direction of the last 15 minutes (`ofi15`) is the hybrid's second most used input at 15 minutes; the leverage effect is standard for equity indices (Black 1976; Glosten, Jagannathan and Runkle 1993); mcsGARCH's daily EGARCH has an asymmetry term but its intraday GARCH does not |
| **X5** `x5_calibration` | the forecast's average level is wrong at some times of day | the forecast at anchor minute i is multiplied by c_i = the mean of y / forecast over the training days' forecasts at that minute (the factor that minimises QLIKE on the training days), re-estimated at every refit from mcsGARCH's in-sample forecasts, exactly the rows the hybrid's correction is trained on; for the last hour one factor (14:30) | one per anchor minute (25, 24 and 1) | the time-of-day group carries 23% and 25% of the gain; LightGBM always learns an overall level for its correction; the hybrid's gains vary strongly by anchor (09:15, 09:30, 15:00) |

**The combination** (`combination`): X1 to X4 in one model (HAR daily level; intraday recursion with overnight
decay, GARCH-X and leverage terms, all estimated jointly by maximum likelihood), then X5's calibration applied to
its forecasts. Nothing else.

**Exploratory leave-one-out variants** (`comb_no_x1` to `comb_no_x5`, Nifty only): the combination without one
extension each (X1 removed means the EGARCH daily level; X5 removed means no calibration).

**Details fixed now.**

* Intraday recursion with all terms (`niftygex.volatility.ext_q2`): within a day, q_{t,i} = omega + alpha
  x_{t,i-1}^2 + gamma_l x_{t,i-1}^2 1{eps_{t,i-1} < 0} + gamma_x z_{t,i-1} + beta q_{t,i-1}; across the night,
  q_{t,0} = qbar + delta (carry - qbar), where carry is the same expression at the last minute of day t-1 and
  qbar = omega / (1 - pi), pi = alpha + beta + 0.5 gamma_l + gamma_x (the persistence, kept below 1). Terms that
  an extension does not switch on are 0 (delta = 1).
* Estimation: Student-t likelihood of the normalised residuals x = eps / sqrt(h s), every minute of the training
  days, L-BFGS-B on unconstrained transforms, from two registered starting points (`EXT_STARTS`: a generic one and
  one next to plain mcsGARCH); the higher likelihood is kept. X1 alone uses the committed GARCH(1,1) code with the
  HAR daily level.
* Forecast of a window: the expected path of the recursion, with E[x^2] = q for future minutes, E[1{eps < 0}
  x^2] = 0.5 q (symmetric errors) and today's running mean updated with the expected x^2; times h_t and the
  diurnal factors, as in mcsGARCH. Windows are cut at the close, as in mcsGARCH.
* With every flag off, `McsGarchExt` runs `McsGarch`'s own code, so the committed forecasts do not change (Section
  11 checks this on real data, a test checks it on simulated data).

## 2. Design

**What is refitted.** Exactly the committed walk-forward of mcsGARCH (`results/pipeline_nse_v2/run_info.json`,
`run_walkforward`): the first 150 days are training only; then before every block of 21 trading days every
model is refitted on all earlier days (days 0 to start - 1), filtered over all days, and forecasts the block. The
daily EGARCH is `daily_egarch_variance(daily returns, train_size=start)`, as committed. The forecast rows are those
of `features.build_dataset` (every 15 minutes from 09:15; windows of 15 and 30 minutes; the last hour trained on
every 60-minute window and evaluated at 14:30) and the target y is computed as there. The driver is
`experiments/p109_mcsgarch_extensions.py run`, one job per refit block, 2 worker processes.

**The hybrid** is not refitted: its committed forecasts (`hybrid_base` in `forecasts_<target>.csv` of the three
committed runs, headline seed set 7) are used as they are.

**Days.** Nifty: 942 days (2 January 2023 to 2 November 2026), 38 refits, **792 test days** (10 August 2023 to
2 November 2026; 19,800 forecasts at 15 minutes, 19,008 at 30 minutes, 792 in the last hour). Bank Nifty: 810 days
to 22 April 2026, 660 test days. Fin Nifty: 754 days to 27 January 2026, 604 test days. The script refuses to run
if the settings, the number of days or the last day differ from the committed run of that index.

**What is run.**

| Index | Targets | Variants |
|---|---|---|
| Nifty | next 15, next 30, last hour | `mcsgarch`, X1 to X5, `combination`, `comb_no_x1` to `comb_no_x5` (12; 10 distinct fits per refit) |
| Bank Nifty, Fin Nifty | next 15, next 30 (also the last hour is written, not analysed) | `mcsgarch`, X1 to X5, `combination` (7; 6 distinct fits) |

Nothing is tuned for Bank Nifty or Fin Nifty: the same code, the same starting values, the same rules.

## 3. Statistics

All losses are QLIKE, y/f - log(y/f) - 1, averaged within each day first (`niftygex.evaluation.daily_losses`);
every day has the same number of forecasts (checked).

* **Gain** of model m: G_m = 100 x (L_mcs - L_m) / L_mcs, in points of mcsGARCH's mean daily QLIKE (positive:
  better than mcsGARCH). The hybrid's gain is G_hyb (5.80 and 7.65 points on Nifty).
* **Share of the hybrid's gain recovered**: S_m = G_m / G_hyb = (L_mcs - L_m) / (L_mcs - L_hyb), with a 95%
  stationary-bootstrap percentile interval (days resampled jointly for the three models in blocks of mean length
  10, 10,000 draws; `stationary_bootstrap_indices` and `bootstrap_means` from `experiments/referee_forecast.py`;
  seeds fixed in the script). S_m can exceed 1 (better than the hybrid) or be negative.
* **Test**: `diebold_mariano(L_worse_daily, L_better_daily, alternative="greater")` from `niftygex.evaluation`, the
  project's standard: Newey-West variance, Harvey-Leybourne-Newbold factor, Student t with n - 1 degrees of
  freedom, one-sided.
* Also reported (descriptive): the gain of every model at every anchor minute and each anchor's share of the total
  QLIKE reduction; gains and shares by expiry regime and on expiry days; the share of the `mcsinfo` hybrid's gain
  (from P106's committed daily losses) and of the seed-averaged hybrid's gain (P99's 21 seed sets); every refit's
  parameters (delta, gamma_l, gamma_x, alpha, beta, nu, the HAR coefficients, the calibration factors) and their
  averages; the sum of the five single shares against the combination's share.

## 4. Hypotheses, tests and decision rules

**Primary family (2 tests, Holm over both):**

* **P1_next15_combination, P1_next30_combination** (Nifty, all 792 test days). H1: the combination has lower mean
  QLIKE than mcsGARCH. One-sided DM. Reported with the share S_comb and its 95% interval.

**Decision rule** (fixed now; applied by `primary_verdict` in the script):

* **Supported, "recovers most of the gain"**: both primary rows have Holm-adjusted p below 0.05 **and** the lower
  end of the 95% interval of S_comb is at least 0.5 at both horizons.
* **Supported in the point estimate only**: both primary rows pass and S_comb is at least 0.5 at both horizons,
  but an interval reaches below 0.5. We then say "about half or more, imprecisely".
* **Contradicted, "recovers less than half"**: the upper end of the 95% interval of S_comb is below 0.5 at both
  horizons (whether or not the combination beats mcsGARCH, which is stated separately).
* **Inconclusive**: anything else, described as found.

**What would contradict the hypothesis** that a few parametric extensions explain most of the gain: the
"contradicted" outcome above. It would mean that the learnt correction does something these five changes do not.

**Reading the five extensions** (fixed now; it changes no verdict):

* An extension is said to **fix a misspecification that matters** if, alone, it beats mcsGARCH (its S1 row,
  Holm-adjusted p below 0.05) at both horizons; "at one horizon" if at one.
* Its size is its single share S_x. Whether it still adds once the others are in is its leave-one-out row (E1,
  exploratory).
* The plain-language answer names the extensions that fix a misspecification that matters, largest single share
  first, and says where in the day their gain lies (S4, E2).

**What we expect** (stated in advance; it changes no rule): the combination beats mcsGARCH at both horizons and
recovers between one third and two thirds of the hybrid's gain; X1 and X2 are the largest single extensions, with
most of their gain at 09:15 and 09:30; X5 gives a smaller gain spread over the day; X3 and X4 are small. The
hybrid still beats the combination (S2). This is a guess, not a rule.

**Secondary families** (each with its own Holm adjustment; they cannot change the primary verdict):

* **S1 (10 tests)**: each extension alone beats mcsGARCH, at 15 and 30 minutes (`S1_<target>_<x>`).
* **S2 (2 tests)**: the hybrid beats the combination (`S2_<target>_hybrid_vs_combination`; estimate: the hybrid's
  gain minus the combination's, points). Not significant would mean that the hybrid and the combination cannot be
  told apart on these days.
* **S3 (2 tests), last hour** (14:30 forecast): the combination beats mcsGARCH; the hybrid beats the combination.
* **S4 (4 tests), the gain by anchor**: the combination beats mcsGARCH at the 09:15 anchor only and at the other
  anchors, at 15 and 30 minutes (`S4_<target>_0915`, `S4_<target>_rest`), each with the share of the hybrid's gain
  on the same anchors.
* **S5 Bank Nifty and S6 Fin Nifty (4 tests each)**, nothing retuned: the combination beats mcsGARCH; the hybrid
  beats the combination; at 15 and 30 minutes, with shares.

**Robustness** (Holm within each family): **R1** the primary rows without the 42 test days of the two refits in
which the committed daily EGARCH fell back to an EWMA (X1 does not use the EGARCH, so these days could favour it);
**R2** the primary rows on the 729 test days before the closing auction (3 August 2026); **R3** S2 with the
hybrid's daily QLIKE averaged over P99's 21 seed sets (`results/seed_bootstrap/daily_losses_<target>.csv`).

**Exploratory** (no adjustment): **E1** leave one extension out of the combination (15 rows: 5 extensions x 3
targets); **E2** each extension alone at 09:15 and at the other anchors (20 rows); **E3** expiry days (169): the
combination against mcsGARCH and the hybrid against the combination; **E4** each extension alone in the last hour;
**E5** P106's `mcsinfo` hybrid against the combination; **E6** each extension alone on Bank Nifty and Fin Nifty
(20 rows).

Every p-value goes in `results/p109_mcsgarch_extensions/tests.csv` with the columns of P106 (test_id, hypothesis,
role, sample, estimate, statistic, p_raw, sidedness, p_method, p_adjusted, adjust_method). Any test added after this
commit is labelled "post hoc".

## 5. The exact commands

From the repository root, at the commit that adds this file, with the raw data listed in `data/MANIFEST.csv` in
`data/raw`:

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py run --index NIFTY --workers 2 \
    > results/p109_mcsgarch_extensions/run_stdout_nifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py run --index BANKNIFTY --workers 2 \
    > results/p109_mcsgarch_extensions/run_stdout_banknifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py run --index FINNIFTY --workers 2 \
    > results/p109_mcsgarch_extensions/run_stdout_finnifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py analyze --plan-commit <this commit> \
    > results/p109_mcsgarch_extensions/analyze_stdout.txt 2>&1
```

`run` writes one file of forecasts and one of parameters per refit block to `data/processed/p109/` (git-ignored; a
finished block is not recomputed if the run must be restarted) and the SHA-256 of every raw input. `analyze` writes
`tests.csv`, `tests_detail.csv`, `verdict.csv`, `extensions.csv`, `gains.csv`, `shares.csv`, `gain_by_anchor.csv`,
`reproduction_check.csv`, `params_by_refit.csv`, `params_summary.csv`, `calibration_factors.csv`,
`daily_losses_<index>_<target>.csv` (each model's daily QLIKE, from which every number can be recomputed),
`timing.csv` and `run_info.json` (plan commit, specifications, input hashes, run time, versions).

## 6. Checks inside the run

* The rebuilt mcsGARCH and y must equal the committed forecasts exactly, on every row of every target and index;
  `analyze` refuses to run otherwise (unless told to report a failure, which would be a deviation).
* P106's committed daily QLIKE of mcsGARCH and P99's seed-7 hybrid daily QLIKE must equal this study's (reported in
  `reproduction_check.csv`).
* Every day has the same number of forecasts.

## 7. What will be reported, whatever the result

In `docs/P109_MCSGARCH_EXTENSIONS.md`, for a beginner reader, every number labelled REAL DATA and taken from a file
in `results/p109_mcsgarch_extensions/`: the plain-language answer first (what the hybrid's correction fixes, as far
as these five extensions can tell); the primary verdict; the table of every model's gain and share; the estimated
parameters in plain words (for example how much of yesterday's close survives the night); the anchor, last-hour,
regime and other-index results; the robustness and exploratory rows with their labels; the reproduction checks; the
run time; a "Deviations from the plan" section; caveats; and how to reproduce.

## 8. Deviations policy

* After this file is committed the design does not change. Any change is a deviation, listed with its reason and
  with whether any outcome had been seen when it was decided.
* A crash is fixed with the smallest change that lets the run finish, and only if it changes no number of a block
  that already finished.
* If a fit fails to converge, that is reported (`converged` in `params_by_refit.csv`) and its forecasts are used as
  they are; no extra starting point is added after outcomes are seen.
* If a job cannot be run (time, crash), the rows that need it are reported as "not computed" with the reason. Bank
  Nifty and Fin Nifty are run after Nifty; if they cannot be finished by 17:00 UTC on 3 November 2026 they are
  dropped and that is reported.

## 9. Known weaknesses

1. **Same days, chosen after P106.** The five extensions were chosen knowing P106's attribution, the anchor results
   and the feature importances, all computed on the same 792 test days. This study explains the hybrid's gain on
   those days; it is not an independent confirmation, and a model chosen this way would need new data before it
   could replace mcsGARCH as a headline.
2. **Five extensions are not all possible ones.** A "contradicted" result means these five do not recover most of
   the gain, not that no parametric model could. Not tried: the overnight price gap (a new piece of information
   that neither mcsGARCH nor the hybrid uses), a weekday-specific diurnal curve (P106 found the calendar inputs add
   nothing), a diurnal curve from a recent window only (the hybrid has no input that tells it the date apart from
   the calendar group, which adds nothing), a two-component intraday GARCH, and realized-GARCH measurement
   equations.
3. **The calibration X5 is fitted in-sample on the training days**, like the hybrid's correction. It is the least
   structural of the five: if it carries the gain, the answer is "mcsGARCH's level is off at these times of day",
   and the factors show where, but not why.
4. **Interactions.** Shares of single extensions need not add up to the combination's share; the leave-one-out rows
   are exploratory.
5. **Local optima.** The extended likelihood is maximised from two starting points; a better maximum might exist.
   The likelihood is nested (each extension contains mcsGARCH), so the fitted likelihood is reported and checked
   against plain mcsGARCH's on simulated data.
6. **The hybrid is one seed set**; R3 repeats S2 with P99's seed average. The extensions themselves are
   deterministic.
7. **The benchmark's EWMA fallback** (two early refits, 42 test days) makes the committed mcsGARCH stronger than a
   real-time rule would. X1 replaces the EGARCH and so does not have it; R1 leaves those days out.
8. **Multi-step forecasts with the leverage term** assume symmetric errors (E[1{eps < 0} x^2] = 0.5 q).

## 10. Code

* `src/niftygex/volatility.py`, new code after the existing code, nothing above changed: `har_daily_variance` (X1),
  `ext_q2` (the extended recursion), `fit_garch_ext` (its likelihood), `McsGarchExt` (a subclass of `McsGarch`
  whose `fit`, `filter` and `forecast_variance` are `McsGarch`'s own when every flag is off).
* `experiments/p109_mcsgarch_extensions.py`, new: `run` (one job per refit block, 2 worker processes, `fork` with
  `maxtasksperchild=1`, as in P106), `check` (the rebuilt mcsGARCH and y against the committed forecasts, no loss),
  `analyze` (Sections 3 and 4). `--synthetic` runs all stages on a simulated market, with the hybrid fitted by the
  pipeline's own walk-forward on it (labelled SYNTHETIC).
* `tests/test_p109_mcsgarch_extensions.py`, new, synthetic data only (13 tests): the extended recursion against a
  plain loop and against mcsGARCH's; `McsGarchExt` with flags off equals `McsGarch` bit for bit; the multi-step
  forecast against the closed form and against a Monte Carlo of the full model; parameter recovery (true alpha 0.04,
  beta 0.93, gamma_l 0.03, gamma_x 0.01, delta 0.3, nu 7) and a likelihood at least plain GARCH's; the HAR level
  (OLS, no look-ahead, scale only in the constant); the calibration (definition, training rows only,
  QLIKE-optimal); the driver's rows and target equal `build_dataset`'s; the driver reproduces `run_walkforward`'s
  mcsGARCH exactly; the variant specifications (each extension and each leave-one-out differs in exactly one
  switch); the share and its interval; the decision rule; and an end-to-end dry run of all three stages.

## 11. Checks run before this file was committed (3 November 2026)

* **The committed mcsGARCH reproduces exactly through this study's driver** (REAL DATA, forecasts and targets only,
  no loss computed). The driver was run with the baseline variant only, into a scratch folder:

  ```
  PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py run --index NIFTY \
      --variants mcsgarch --workers 2 --work <scratch folder>
  PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py check --indexes NIFTY --work <scratch folder>
  ```

  Nifty: **every forecast and every target is identical to the last bit**: 19,800 of 19,800 at 15 minutes,
  19,008 of 19,008 at 30 minutes and 792 of 792 in the last hour, for `y` and `mcsgarch` (largest relative
  difference 0). The 38 baseline refits took 1.3 minutes on 2 cores. The same check with `--index BANKNIFTY` and
  `--index FINNIFTY` (`check --indexes BANKNIFTY FINNIFTY`): Bank Nifty 16,500 of 16,500, 15,840 of 15,840 and 660
  of 660 identical; Fin Nifty 15,100 of 15,100, 14,496 of 14,496 and 604 of 604 identical (`y` and `mcsgarch`,
  largest relative difference 0); about 1 minute each. In all three runs every setting and the number and range of
  days matched the committed `run_info.json`.
* **Dry run** (SYNTHETIC: 200 simulated days, 7 refits, two targets): all three stages ran; the rebuilt mcsGARCH
  equalled the pipeline's walk-forward on every row (1,260 and 1,120 forecasts); stage 1 took 0.5 minutes.
* **Speed** (SYNTHETIC, 900 days x 375 minutes, the size of the last Nifty refit): one fit of the full extended
  model took 7.8 s and one-extension fits about 3 s, against 1.0 s for plain mcsGARCH. With 10 fits per refit and
  38 refits, we expect stage 1 to take about 15 to 30 minutes for Nifty and 10 to 20 minutes for each of the other
  two indices on 2 cores.
* **Tests.** `PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_p109_mcsgarch_extensions.py`: 13
  passed. The full suite (`python -m pytest -q`): 607 passed, none failed, in about 13 minutes.
