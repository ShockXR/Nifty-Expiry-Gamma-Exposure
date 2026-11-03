# P106 pre-registration: where does the forecasting hybrid's gain come from?

**Status.** Written on 3 November 2026 (phase 1) by an AI agent (Claude) working for Shreyan Thanki. **This design
is committed to git before any outcome of this study has been computed.** No forecast of any ablated hybrid (a
hybrid with a feature group removed) has been made on real data, and no loss, margin, contribution, test statistic
or bootstrap draw of this study has been computed on real data. The only real-data computation so far is the
reproduction check in Section 11: the full hybrid with the headline's seeds was refitted through this study's new
code path and its forecasts were compared with the committed ones, forecast by forecast, without computing any
loss (those forecasts and their losses are public in `results/pipeline_nse_v2/`). The script and its tests were
also run end to end on a simulated market (SYNTHETIC). The commit that adds this file is the time stamp; phase 2
(the run) starts only after that commit exists and has been pushed. The code that phase 2 runs
(`experiments/p106_gain_decomposition.py`, one new code path in `src/niftygex/forecast.py`) and its tests
(`tests/test_p106_gain_decomposition.py`, synthetic data only) are in the same commit. Phase 2 writes
`results/p106_gain_decomposition/` and `docs/P106_GAIN_DECOMPOSITION.md`. All results will be REAL DATA.

## In plain words

Our hybrid model forecasts how much the Nifty will move over the next 15 or 30 minutes. It takes the anchor
paper's model (mcsGARCH) and multiplies its forecast by a correction learnt by LightGBM, a machine-learning model.
The correction looks at 41 inputs ("features"): the previous evening's option prices, India VIX, the time of day,
the expiry calendar and recent realized volatility. The hybrid's loss (QLIKE) is 5.8% and 7.6% lower than
mcsGARCH's. Against a fairer benchmark, an mcsGARCH whose daily level comes from the option market
(`mcsgarch_iv`), it is 4.8% and 5.9% better, but only since SEBI cut weekly options in November 2024; before
that the two were almost equal (0.2% and 0.7%).

This study asks two questions:

1. **Which inputs carry the gain?** We sort the 41 inputs into five groups and refit the hybrid five times, each
   time without one group, changing nothing else. If the loss goes up when a group is removed, that group carried
   part of the gain. We test this with the project's usual Diebold-Mariano test.
2. **Why does the edge over the fairer benchmark appear only since November 2024?** We split it into "the
   benchmark got worse" and "the hybrid got better", and we test, group by group, whether a group's contribution
   to the hybrid was different before and after the cut.

LightGBM uses random numbers, so two fits that differ only in their seeds give slightly different results. Small
differences between variants could be just that. We therefore repeat the main comparisons with four more sets of
seeds and call a group's contribution real only if every seed set agrees on its sign.

## Why this study

The hybrid is the project's forecasting headline (`docs/MODEL_ITERATIONS_REAL.md`), and the referee response
(`docs/REFEREE_RESPONSE_FORECAST.md`, M5 and priority 5) narrowed it: at 15 and 30 minutes it survives an
information-matched benchmark, but its edge over that benchmark comes from the periods since November 2024. Nobody
has yet asked which of the hybrid's inputs produce the gain, or why the matched edge appears when it does. A
reader of the paper will ask both.

## Prior knowledge (what was known when this was written)

All REAL DATA, from committed files. Margins are in % of mcsGARCH's QLIKE (negative: better than mcsGARCH).

* **The headline** (`results/pipeline_nse_v2/scoreboard_*.csv`, 792 test days, 10 August 2023 to 2 November
  2026): `hybrid_base` -5.80% (15 minutes), -7.65% (30 minutes), -6.36% (last hour); `mcsgarch_iv` -1.05%,
  -1.88%, -1.24%.
* **Against the matched benchmark** (`results/referee_forecast/fair_benchmarks.csv`, `gw_group_margins.csv`):
  -4.8% (p 0.005) and -5.9% (p below 0.001) on all days; by regime, in % of `mcsgarch_iv`'s QLIKE:

  | Horizon | multi-weekly (to 19 Nov 2024, 312 days) | Thursday weekly (192 days) | Tuesday weekly (288 days) |
  |---|---|---|---|
  | next 15 | -0.22% | -7.30% | -6.96% |
  | next 30 | -0.70% | -8.21% | -9.29% |

  The Giacomini and White regime test (two regime dummies) gave p 0.100 and 0.027, Holm-adjusted 1.000 and 0.652
  (`gw_tests.csv`).
* **The same regimes, each model against mcsGARCH** (`scoreboard_*.csv`):

  | Horizon, model | multi-weekly | Thursday weekly | Tuesday weekly |
  |---|---|---|---|
  | next 15, `mcsgarch_iv` | -4.98% | +2.31% | +0.14% |
  | next 15, `hybrid_base` | -5.19% | -5.16% | -6.83% |
  | next 30, `mcsgarch_iv` | -6.06% | +1.99% | -0.49% |
  | next 30, `hybrid_base` | -6.70% | -6.42% | -9.73% |

  Simple arithmetic on these committed numbers already says where most of the change comes from: in points of
  mcsGARCH's QLIKE, the hybrid's edge over `mcsgarch_iv` is (`mcsgarch_iv`'s extra loss over mcsGARCH) plus (the
  hybrid's gain over mcsGARCH). At 15 minutes that is -4.98 + 5.19 = 0.21 in the multi-weekly period and
  2.31 + 5.16 = 7.47 and 0.14 + 6.83 = 6.97 in the two later regimes; at 30 minutes -6.06 + 6.70 = 0.64, then
  1.99 + 6.42 = 8.41 and -0.49 + 9.73 = 9.24. **So the benchmark lost about 5 to 8 points of its advantage over
  mcsGARCH after the cut, while the hybrid's gain over mcsGARCH changed by 0 to 3 points.** This study reports
  that split formally (S2 below), but it is not a new finding, and the plan says so.
* **The option-implied and new calendar features as one block** (`ablation_<target>.csv`, family "base
  upgrades": `hybrid_v1`, which lacks the implied group and six calendar features, against `hybrid_base`; change
  in % of `hybrid_v1`'s QLIKE, negative = the block helps; one-sided p): all days -0.29% (p 0.23) at 15 minutes and
  +0.17% (p 0.65) at 30 minutes; multi-weekly -0.66% (p 0.095) and -1.38% (p 0.038); Thursday weekly +0.87% and
  +1.31%; Tuesday weekly -0.78% and +1.00%. So the block helped a little before the cut and not after. `hybrid_v1`
  is not one of this study's variants (it removes the implied group and part of the calendar group at once), but
  this result is the main reason for the direction we state for the implied group in Part 2.
* **Feature importance** (`feature_importance_<target>.csv`, `hybrid_base`, share of LightGBM's split gain
  averaged over refits), summed over this study's groups: next 15 minutes realized 49.6%, implied 30.0%, time of
  day 16.8%, calendar 2.1%, VIX 1.5%; next 30 minutes 47.3%, 32.7%, 15.7%, 2.6%, 1.8%. Gain importance says what
  the trees split on, not what they would lose without it (another group can stand in), which is why we refit.
* **Other models** (`docs/MODEL_ITERATIONS_REAL.md`, Iterations 3 and 5): models that lean on the straddle's level
  (`mcsgarch_iv`, the ridge log-HAR) did well before November 2024 and lost their edge after it; on expiry days
  realized variance was 0.57 and 0.61 of the straddle-implied variance in the two earlier sub-periods but 0.47
  after 20 November 2024 (`dev_implied_checks.csv`, development window).
* **Seed noise** (P99, `results/seed_bootstrap/summary.csv`): the standard deviation of the hybrid's margin over 20
  five-seed ensembles is 0.075 (15 minutes), 0.112 (30 minutes) and 0.244 points (last hour) on all days. If the
  seed noise of two variants were independent, the difference between them would have a seed SD of about 1.4
  times that: about 0.11, 0.16 and 0.35 points.
* **Bank Nifty and Fin Nifty** (P91, P104): the same hybrid beats mcsGARCH there too (3.9% and 5.0%; 4.9% and
  6.1% at 15 and 30 minutes). No ablation by feature group has been run on any index.

## 1. Feature groups

Every one of the 41 columns that `hybrid_base`'s LightGBM correction sees (the 37 of `BASE_FEATURES` in
`src/niftygex/features.py` plus the 4 that `run_walkforward` adds for every "base" model) is in exactly one group.
A test checks this.

| Group | Columns | Why there |
|---|---|---|
| **implied** (16): everything computed from the option chain published at the previous close | `log_impvar_day`, `log_impvar_to_exp`, `log_iv_m21`, `log_fwdvar_day`, `exp_premium`, `term_slope`, `skew_near`, `smile_near`, `skew_m21`, `log_pcr_oi_near`, `log_pcr_vol_near`, `log_pcr_oi_all`, `vrp_w`, `abs_ret_today_sd`, `log_mcs_iv`, `log_implied_window` | the 14 `IMPLIED_FEATURES` and the two columns built from the straddle-implied variance (`mcsgarch_iv`'s forecast and the implied variance spread over the window). `vrp_w` (implied minus realized variance) and `abs_ret_today_sd` (the move since the open in implied standard deviations) go here because each carries the implied level: with them left in, removing the group would not remove the option market's information. |
| **vix** (1) | `log_vix` | India VIX at the previous close, a 30-day option-implied measure published by NSE, kept apart to see whether it and the chain stand in for each other |
| **intraday** (4): time of day | `minute_frac`, `mins_left`, `exp_x_minute`, `log_s_window` | where the window sits in the session; `exp_x_minute` is the expiry-day version; `log_s_window` is mcsGARCH's diurnal share of the window, a function of the time of day and the expiry flag only |
| **calendar** (8): calendar and regime | `is_expiry`, `tdte`, `regime`, `weekday`, `is_monthly_expiry`, `day_before_expiry`, `bn_expiry_only`, `closing_auction` | dated rules known in advance |
| **realized** (12): realized-variance history | `log_rv_back_h`, `log_rv_back_15`, `log_rv_back_60`, `log_rv_today_pm`, `log_rv_d1`, `log_rv_w`, `log_rv_m`, `log_rv_prevday_same`, `jump_share_d1`, `log_vol15_rel`, `ofi15`, `log_mcs` | everything computed from past one-minute index returns. `ofi15` is a signed-return proxy (the bars have no volume) and `log_vol15_rel` is constant for the same reason. `log_mcs`, mcsGARCH's own forecast, is a function of past returns, the time of day and the expiry flag; it is placed here because it is the benchmark's summary of past returns. Removing it as a feature does not remove it as the hybrid's starting point: the correction is always applied to mcsGARCH's forecast. |

**Variants** (each is `hybrid_base` refitted without the listed columns; model name `hybrid_drop_<variant>`):

| Variant | Columns removed | Role |
|---|---|---|
| `full` | none (this is `hybrid_base`) | reference; must reproduce the committed forecasts exactly |
| `implied`, `vix`, `intraday`, `calendar`, `realized` | one group | primary (Part 1 and Part 2) |
| `optionmkt` | implied and vix together (17 columns) | secondary: all option-market information at once, since the chain and VIX can stand in for each other |
| `mcsinfo` | implied, vix and the calendar group except `is_expiry` (24 columns) | secondary: a hybrid that sees only what mcsGARCH itself uses (past one-minute returns, the time of day and the expiry-day flag, which sets mcsGARCH's expiry-day diurnal curve). It separates "method" from "information". |
| `implied_level` | `log_impvar_day`, `log_impvar_to_exp`, `log_iv_m21`, `log_fwdvar_day`, `vrp_w`, `abs_ret_today_sd`, `log_mcs_iv`, `log_implied_window` | exploratory: the level part of the implied group |
| `implied_shape` | `exp_premium`, `term_slope`, `skew_near`, `smile_near`, `skew_m21`, `log_pcr_oi_near`, `log_pcr_vol_near`, `log_pcr_oi_all` | exploratory: term-structure shape, skew, smile and put/call ratios |

## 2. Design

**What is refitted.** Exactly the walk-forward of the committed runs (`results/pipeline_nse_v2/run_info.json`):
the pipeline's own `load_market` (from `scripts/run_pipeline.py`), `build_gex_panel`, `build_implied_panel`,
`build_dataset` and `run_walkforward`, with the committed `WalkForwardConfig` (150 training days, refit before every
block of 21 trading days on all earlier days, mcsGARCH refitted every 21 days, LightGBM default hyperparameters
with the number of boosting rounds picked inside each training window, 5-seed averaging, last hour trained on every
60-minute window and evaluated at 14:30). Each walk-forward fits `mcsgarch`, `mcsgarch_iv` (both deterministic)
and up to five hybrid variants side by side; the variants share the mcsGARCH fits and nothing else (a test checks
that the full hybrid is identical whatever is fitted next to it). The script refuses to run if any setting, the
number of days or the last day differs from the committed run of that index.

**The only code change.** `src/niftygex/forecast.py` gets a new optional field `WalkForwardConfig.drop_sets`
(empty by default) and a new feature-set name `drop_<name>` in `model_columns`: the version 2 base set with its
extra columns, minus `drop_sets[name]`, in the same order. No default and no existing model changes.

**Seeds.** The headline's base seed 7 (LightGBM seeds 7, 1007, 2007, 3007 and 4007; the base seed also seeds the
pass that picks the number of rounds) for every variant, target and index. **Seed repeats:** base seeds 101, 102,
103 and 104 (the first four of P99's twenty new ensembles) for `full`, the five groups, `optionmkt` and `mcsinfo`,
at 15 and 30 minutes on Nifty. The `full` refits with these seeds must reproduce P99's committed daily losses
(`results/seed_bootstrap/daily_losses_<target>.csv`).

**What is run** (Nifty: 942 days, 2 January 2023 to 2 November 2026, 792 test days):

| Index | Targets | Seeds | Variants | Variant fits |
|---|---|---|---|---|
| Nifty | next 15, next 30, last hour | 7 | all 10 | 30 |
| Nifty | next 15, next 30 | 101, 102, 103, 104 | `full`, 5 groups, `optionmkt`, `mcsinfo` | 64 |
| Bank Nifty (810 days to 22 April 2026, 660 test days) | next 15, next 30 | 7 | `full`, 5 groups | 12 |
| Fin Nifty (754 days to 27 January 2026, 604 test days) | next 15, next 30 | 7 | `full`, 5 groups | 12 |

Jobs run in this order (Nifty headline seed first). Bank Nifty and Fin Nifty are optional secondary work: if they
cannot be finished by 10:30 UTC on 3 November 2026, they are dropped and that is reported as a deviation; nothing
in Part 1 or Part 2 depends on them.

**Days and periods.** All test days (792 on Nifty). **Period 1** = the multi-weekly regime, test days 10 August
2023 to 19 November 2024 (312 days); **period 2** = from 20 November 2024, the day SEBI's cut took effect, to 2
November 2026 (480 days: 192 Thursday-weekly and 288 Tuesday-weekly). This is the pipeline's own regime label
(regime code 0 against 1 or 2); the script checks that the date split and the label agree on every day.

## 3. Statistics

All losses are QLIKE, y/f - log(y/f) - 1, averaged within each day first (`niftygex.evaluation.daily_losses`).
Every day has the same number of forecasts (25 at 15 minutes, 24 at 30 minutes, 1 in the last hour; checked), so
the mean of daily means is the mean over all forecasts, as in the scoreboards.

**Part 1, contribution of a group.** With L_full, L_v and L_mcs the mean daily QLIKE of the full hybrid, the
variant without group v, and mcsGARCH:

* contribution C_v = 100 x (L_v - L_full) / L_mcs, in **points of mcsGARCH's QLIKE** (positive: removing the group
  raises the loss, so the group helps). The hybrid's gain is G = 100 x (L_mcs - L_full) / L_mcs (5.80 and 7.65
  points on all days), so C_v and G are in the same unit;
* **share of the gain** carried by the group, C_v / G, with a 95% stationary-bootstrap percentile interval (days
  resampled in blocks of mean length 10, 10,000 draws; `stationary_bootstrap_indices` and `bootstrap_means` from
  `experiments/referee_forecast.py`). Shares need not add up to 100%: groups that carry the same information
  (for example the chain and VIX) each look unimportant when removed alone, and groups that work together can
  each look large;
* **test**: `diebold_mariano(L_v_daily, L_full_daily, alternative="greater")` from `niftygex.evaluation`: Newey-West
  variance, Harvey-Leybourne-Newbold factor, Student t with n - 1 degrees of freedom, one-sided (removal raises the
  loss), the project's standard DM test.

**Part 2, the two periods.** Within each period k, everything is in points of that period's mean daily QLIKE of
mcsGARCH (L_mcs,k), which makes the pieces add up:

* the hybrid's gain over mcsGARCH, G_k = 100 x (L_mcs,k - L_full,k) / L_mcs,k;
* the benchmark's own change, B_k = 100 x (L_iv,k - L_mcs,k) / L_mcs,k (positive: `mcsgarch_iv` is worse than
  mcsGARCH);
* the hybrid's edge over `mcsgarch_iv`, E_k = 100 x (L_iv,k - L_full,k) / L_mcs,k = B_k + G_k. (Reported also in %
  of `mcsgarch_iv`'s QLIKE, 100 x (L_full,k / L_iv,k - 1), the unit of `gw_group_margins.csv`.);
* each group's contribution, C_v,k = 100 x (L_v,k - L_full,k) / L_mcs,k, so that the hybrid without group v has
  gain G_k - C_v,k and edge E_k - C_v,k.

The change between the periods is D = (value in period 2) - (value in period 1), for E, B, G and each C_v. **Test**
(two-sided): each period's value is a ratio of two means; its variance is found by the delta method (influence
series 100 x (num_t - r_k x den_t / 100) / mean(den) within the period) with the Newey-West long-run variance and
the DM test's bandwidth rule, floor(4 (n_k / 100)^(2/9)); the two periods are treated as independent, so
Var(D) = Var1 + Var2; t = D / sqrt(Var(D)) is compared with a Student t with n - 2 degrees of freedom (n = 792).
Two-sided because theory points both ways for most groups (Section 4). Robustness: the same D with a stationary
bootstrap of days within each period (mean block 10, 10,000 draws), two-sided p = (1 + #{|D* - D| >= |D|}) /
10,001. Also reported: the share of the change in the edge that is the benchmark's, D_B / D_E, with a bootstrap
interval; and, descriptively, every quantity in each of the three regimes.

**Seed repeats.** For each primary row, the same estimate (C_v for Part 1, D for Part 2) in each of the five seed
sets (7, 101, 102, 103, 104): the number of seed sets with the same sign as the headline seed set (for Part 1: with
C_v > 0), their mean, standard deviation (SD), minimum and maximum.

## 4. Hypotheses, tests and decision rules

**Primary family (20 tests, one Holm adjustment over all 20):**

* **Part 1, rows P1_<horizon>_<group>** (5 groups x next 15 and next 30 minutes, all days): H1: removing the group
  raises the hybrid's QLIKE. One-sided DM test.
* **Part 2, rows P2_<horizon>_<group>** (5 groups x 2 horizons): H2: the group's contribution differs between
  period 1 and period 2. Two-sided delta-method test of D.

**Verdict of each primary row** (fixed now):

* **supported**: Holm-adjusted p below 0.05 **and** all five seed sets agree in sign (Part 1: C_v > 0 in all five;
  Part 2: D has the headline's sign in all five);
* **seed-fragile**: Holm-adjusted p below 0.05 but at least one seed set disagrees;
* **not supported**: Holm-adjusted p of 0.05 or more.

Separately, any row whose headline estimate is smaller in absolute value than twice the SD across the five seed
sets is flagged **"within seed noise"**, and its size is not interpreted, whatever its p-value. A group is said to
"carry part of the gain" if its Part 1 row is supported at both horizons, "at one horizon" if at one.

**What we expect, and what would contradict it** (expectations stated in advance; they do not change any rule):

| Row | Expectation | Contradicted if |
|---|---|---|
| P1 realized | supported at both horizons; the largest contribution, more than a third of the gain. The trees use recent intraday realized variance to correct mcsGARCH's slow daily level. | not supported at either horizon |
| P1 intraday | supported at both horizons, 0.5 to 2 points (16% to 17% of gain importance; mcsGARCH's diurnal curve is one average shape) | not supported at either horizon |
| P1 implied | not supported, or below 0.5 points: VIX and the realized history stand in (the committed implied-plus-calendar block was worth -0.29% and +0.17%) | supported at both horizons with more than 1 point |
| P1 vix | not supported (the chain stands in) | supported at both horizons |
| P1 calendar | not supported (about 2% of gain importance) | supported at both horizons |
| P2 implied | **direction stated: smaller since the cut (D < 0).** Reasons: the implied-plus-calendar block helped the hybrid only in the multi-weekly period; every model that leans on the straddle's level lost its edge after the cut; and on expiry days the straddle became expensive relative to realized variance (0.47 against 0.57 to 0.61), a shift that a model trained mostly on earlier days learns slowly. The competing view (the implied group helps more after the cut because Nifty's own weekly expiry then dominates option trading) predicts D > 0. | D > 0 and supported at either horizon (that would support the competing view) |
| P2 realized, intraday, vix, calendar | no direction stated | (two-sided; any supported difference is reported as found) |

**Secondary families** (each with its own Holm adjustment; they cannot change a primary verdict):

* **S1 (2 tests), the edge over `mcsgarch_iv` differs between the periods** (D_E, two-sided). Expected: larger in
  period 2. This was already seen on the same data (Prior knowledge); it is a re-test, not a new confirmation.
* **S2 (4 tests), the decomposition**: D_G = 0 and D_B = 0 (two-sided), at each horizon. Expected: D_B large and
  significant, D_G small, so that most of the change in the edge is the benchmark's (D_B / D_E above 0.5).
  Contradicted if the bootstrap interval of D_B / D_E lies wholly below 0.5. Also known in outline from committed
  numbers.
* **S3 (6 tests), information or method**, at 15 and 30 minutes, one-sided DM: (a) removing `optionmkt` raises the
  loss; (b) the `mcsinfo` hybrid beats mcsGARCH (method alone, no extra information); (c) removing what `mcsinfo`
  removes raises the loss (the value of the information mcsGARCH lacks). Expected: (b) yes; (a) and (c) open.
* **S3b (4 tests)**: the Part 2 test for `optionmkt` and `mcsinfo`.
* **S4 (5 tests), last hour**: Part 1 for the five groups (headline seed only; P99's last-hour seed SD is 0.244
  points, so any last-hour contribution below about 0.7 points is within seed noise).
* **S5 and S6 (10 tests each), Bank Nifty and Fin Nifty**: Part 1 for the five groups at 15 and 30 minutes
  (headline seed). Also the Spearman rank correlation of the five contributions with Nifty's (descriptive).
  Expected: the realized group first on every index.

**Robustness** (Holm within each family; descriptive with respect to the verdicts): R1 Part 1 on daily losses
averaged over the five seed sets; R2 Part 1 without the 09:15 forecast (which carries the overnight news and a
large part of the gain); R3 Part 1 on the 729 days before the closing auction; R5 Part 2 with the day-block
bootstrap; R6 S1 and S2 on the subsets of R7 and R8; R7 Part 2 with period 2 cut at the closing auction (3 August
2026); R8 Part 2 without the 42 test days of the two refits in which mcsGARCH's daily EGARCH fell back to an EWMA
(August to October 2023; `docs/REFEREE_RESPONSE_FORECAST.md`, side finding 1); R9 Part 2 on seed-averaged losses.

**Placebo** (Holm within each family): PL1, the full hybrid with seed set 7 against the same hybrid with seed set
101, 102, 103 or 104 (8 one-sided DM tests: what "removing nothing" looks like to the DM test, so the reader can
see how often seed noise alone looks significant); PL2, the Part 2 period test applied to those seed-only
differences (8 tests).

**Exploratory** (no adjustment): E1 `implied_level` and `implied_shape` (Part 1); E2 Part 1 on the 169 expiry days;
E3 Part 2 for `implied_level` and `implied_shape`; E4 `optionmkt`, `mcsinfo`, `implied_level`, `implied_shape` in
the last hour; E5 Part 2 and the edge decomposition in the last hour.

Every p-value goes in `results/p106_gain_decomposition/tests.csv` (columns test_id, hypothesis, role, sample,
estimate, statistic, p_raw, sidedness, p_method, p_adjusted, adjust_method). Any test added after this commit is
labelled "post hoc".

## 5. The exact commands

From the repository root, at the commit that adds this file, with the raw data listed in `data/MANIFEST.csv` in
`data/raw`:

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p106_gain_decomposition.py run --index NIFTY --workers 2 \
    > results/p106_gain_decomposition/run_stdout_nifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p106_gain_decomposition.py run --index BANKNIFTY --workers 2 \
    > results/p106_gain_decomposition/run_stdout_banknifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p106_gain_decomposition.py run --index FINNIFTY --workers 2 \
    > results/p106_gain_decomposition/run_stdout_finnifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p106_gain_decomposition.py analyze --plan-commit <this commit> \
    > results/p106_gain_decomposition/analyze_stdout.txt 2>&1
```

`run` stores each variant's forecasts in `data/processed/p106/` (git-ignored; a finished job is not recomputed if
the run must be restarted) and the SHA-256 of every raw input. `analyze` writes `tests.csv`, `tests_detail.csv`,
`verdict.csv`, `contributions.csv`, `shares.csv`, `periods.csv`, `regimes.csv`, `seed_repeats.csv`,
`other_index_ranking.csv`, `reproduction_check.csv`, `daily_losses_nifty_<target>_s<seed>.csv` (each variant's daily
QLIKE, from which every number can be recomputed), `timing.csv` and `run_info.json` (plan commit, seeds, input
hashes, run time, versions).

## 6. Checks inside the run

* The full hybrid with seed 7 must reproduce the committed `hybrid_base` forecasts of `results/pipeline_nse_v2/`
  (and of the Bank Nifty and Fin Nifty runs) exactly, together with `y`, `mcsgarch` and `mcsgarch_iv`.
* The full hybrid with seeds 101 to 104 must reproduce P99's daily losses.
* `y`, `mcsgarch` and `mcsgarch_iv` must be identical in every work file of a target (the script refuses
  otherwise).

## 7. What will be reported, whatever the result

In `docs/P106_GAIN_DECOMPOSITION.md`, for a beginner reader, every number labelled REAL DATA and taken from a file
in `results/p106_gain_decomposition/`: the verdict first; the share of the gain carried by each group; the
November 2024 split (benchmark against hybrid) and the group-level period tests; the seed repeats; the secondary,
robustness, placebo and exploratory results with their labels; the reproduction checks; the run time; a
"Deviations from the plan" section; caveats; and what it changes in the project's claims.

## 8. Deviations policy

* After this file is committed the design does not change. Any change is a deviation, listed with its reason and
  with whether any outcome had been seen when it was decided.
* A crash is fixed with the smallest change that lets the run finish, and only if it changes no number of a job
  that already finished.
* If a reproduction check fails, that is reported, the cause is looked for, and the results are still given with
  the failure stated next to them.
* If a job cannot be run (time, crash), the rows that need it are reported as "not computed" with the reason.

## 9. Known weaknesses

1. **Leave-one-out measures what a group adds given all the others.** When two groups carry the same information
   (the chain and VIX; the realized history and mcsGARCH's own forecast), each can look useless when removed
   alone. The `optionmkt` and `mcsinfo` variants and the shares help, but a full attribution (for example Shapley
   values over all 32 subsets of groups) is not attempted.
2. **The protocol is held fixed.** Each variant keeps the default hyperparameters and the in-window choice of the
   number of rounds. A variant with fewer columns might do better with other settings; we measure the loss under
   the protocol that produced the headline, not the best possible model without the group.
3. **Part 2 is not independent of what motivated it.** The edge's period pattern (S1) and its split (S2) were
   visible in committed files, and the direction for the implied group was chosen after seeing that the
   implied-plus-calendar block helped only before the cut. The group-level tests are new, but they use the same
   days.
4. **A period is a stretch of time.** A difference between the periods may come from the SEBI change, from the
   longer training window available later, or from anything else that changed in 2025 and 2026. The design cannot
   separate these.
5. **Five seed sets** are enough to see whether a sign is stable, not to estimate the seed SD of a contribution
   precisely (4 degrees of freedom). The last hour and the other indices use the headline seed only.
6. **The benchmark is our implementation of mcsGARCH**, with the EWMA fallback in two refits (R8 leaves those days
   out).
7. **Grouping choices.** `log_mcs` sits in the realized group, `abs_ret_today_sd` and `vrp_w` in the implied group,
   `log_s_window` in the time-of-day group, `is_expiry` in the calendar group. Other choices are defensible; they
   are fixed here.

## 10. Code

* `experiments/p106_gain_decomposition.py`, new: `run` (forecasts, one invocation per index, 2 worker processes,
  `fork` with `maxtasksperchild=1`, as in P99), `check` (the full hybrid against the committed forecasts, no loss),
  `analyze` (Sections 3 and 4). `--synthetic` runs both stages on a simulated market (labelled SYNTHETIC).
* `src/niftygex/forecast.py`: the new `drop_sets` field and `drop_<name>` feature set (Section 2). No default
  changes; `scripts/run_pipeline.py` is not touched.
* `tests/test_p106_gain_decomposition.py`, new, synthetic data only (15 tests): the groups partition the 41 columns;
  the new code path returns the base set minus the group, in order, with no change to existing models; variants
  fitted together do not change the full hybrid, and an empty drop set reproduces it exactly; each variant's
  importance table lists exactly its columns; the configuration matches all three committed runs' `run_info.json`;
  the job plan covers every registered fit once, in order; work files round-trip exactly and the reproduction check
  catches a changed forecast; the date split equals the regime label; the contribution equals its definition and
  the DM test; the period-difference test equals a hand computation, finds a real shift and rejects about 5% of the
  time under the null with autocorrelated, heavy-tailed differences; the bootstrap and the share interval; the
  verdict labels and Holm families; and an end-to-end dry run of both stages.

## 11. Checks run before this file was committed (3 November 2026)

* **The full hybrid reproduces exactly through the new code path.** With the code in this commit, the full hybrid
  (headline seed 7, fitted next to `mcsgarch` and `mcsgarch_iv`) was refitted on Nifty at all three targets,
  writing to a scratch folder:

  ```
  PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p106_gain_decomposition.py run --index NIFTY \
      --variants full --seeds 7 --workers 1 --work <scratch folder>
  PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p106_gain_decomposition.py check --indexes NIFTY --work <scratch folder>
  ```

  The market loaded with 942 days (2 January 2023 to 2 November 2026) and every setting matched
  `results/pipeline_nse_v2/run_info.json`. Compared with the committed `forecasts_<target>.csv`, read back at full
  precision, **every forecast is identical to the last bit**: 19,800 of 19,800 at next 15 minutes, 19,008 of
  19,008 at next 30 minutes and 792 of 792 in the last hour, for `y`, `mcsgarch`, `mcsgarch_iv` and `hybrid_base`
  (largest relative difference 0 in every case). `check` computes no loss, and no loss was computed. No ablated
  variant was fitted on real data.
* **Inputs.** All 1,891 raw files the Nifty run reads (the bars, the official day list, India VIX, 944 bhavcopies
  and 944 participant files) agree with `data/MANIFEST.csv` (hashes as the manifest computes them: of the
  decompressed content for `.gz` files).
* **Run time.** Loading the market took 59 s and building the panels and the three datasets 186 s; the three
  single-variant walk-forwards took 154 s (next 15), 134 s (next 30) and 102 s (last hour) on one core. The study
  fits 118 variants in 30 jobs of up to 5 variants that share one mcsGARCH fit, so we expect 1.5 to 2 hours on 2
  cores in all, Nifty first.
* **Dry run.** Both stages ran end to end on a simulated market (`--synthetic`, SYNTHETIC: 200 days, 2 seed sets,
  2 targets): 8 jobs in 38 s, and `analyze` wrote every file listed in Section 5.
* **Tests.** `PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_p106_gain_decomposition.py`: 15
  passed. In the size test (400 simulated pairs of periods of 160 and 240 days, AR(1) differentials with Student t
  errors and the same mean in both periods) the 5% period-difference test rejected 6.25% of the time (25 of 400;
  the test accepts 2% to 9%), so it is close to its nominal size, if slightly liberal. The existing forecasting tests (`test_forecast.py`, `test_forecast_v2.py`,
  `test_seed_bootstrap.py`) pass with the new code path.
