# P91 pre-registration: does the Nifty forecasting hybrid beat mcsGARCH on Bank Nifty, with nothing changed?

**Status.** Written on 2 and 3 November 2026 (phase 1). **This design is committed to git before any Bank
Nifty outcome of this study has been computed.** This study has computed no Bank Nifty forecast, no forecast
loss, no test statistic and none of its forecast targets (the realized variance of each 15-minute, 30-minute
and last-hour window). Earlier studies in this repository did compute some Bank Nifty realized variances for
other questions; they are listed under Prior knowledge. None of them was an out-of-sample forecast or a
forecast loss. The commit that adds this file is the time stamp, and phase 2 (the run) starts only after that
commit exists. Phase 2 writes `results/pipeline_nse_banknifty/` (every output of the pipeline),
`results/pipeline_nse_banknifty_stdout.txt` (the log) and `docs/BANKNIFTY_FORECAST.md` (the write-up). The
code that phase 2 runs is in the same commit as this file (Section 11). All results will be REAL DATA.

## In plain words

On Nifty, the anchor paper's intraday volatility model (mcsGARCH), corrected by a machine-learning model
(LightGBM), forecast the next 15 and 30 minutes of volatility better than mcsGARCH alone. Every
choice in that model was made while looking at Nifty, so the result may partly fit Nifty's quirks. Here we run
exactly the same model, with no change at all, on Bank Nifty, the other big NSE index, over 660 trading days
(10 August 2023 to 22 April 2026), and compare it with mcsGARCH in the same way.

* If the corrected model wins at both 15 and 30 minutes, and the statistical test still passes after allowing
  for having made two tests, we say the Nifty result **replicates**.
* If it passes at only one of the two, it **partly replicates**. If at neither, it **does not replicate**.
* We also look, without changing the verdict, at the last hour of the day, at Bank Nifty's expiry days, and at
  whether adding dealer gamma (GEX) helps. For GEX we predict, as on Nifty, that it does not.

The three biggest differences cannot be avoided (Section 3 lists all of them). Bank Nifty has no volatility index of its own, so India VIX (built from
Nifty options) stands in. Bank Nifty lost its weekly options in November 2024, so after that its option
features describe a contract up to a month away. And the Bank Nifty minute data end on 22 April 2026, before
NSE's closing auction began, so the sample is shorter. All of this was written down and committed before any
Bank Nifty forecast was made.

## Why this test

The forecasting contest on Nifty (`docs/MODEL_ITERATIONS_REAL.md`, `results/pipeline_nse_v2/`) found that
`hybrid_base`, the anchor paper's mcsGARCH multiplied by a LightGBM correction, has a lower QLIKE loss than
mcsGARCH (refitted every 21 trading days) by 5.8% for the next 15 minutes, 7.6% for the next 30 minutes and
6.4% for the last hour, on 792 test days. Every design choice was made on Nifty data only. The anchor paper
(Chhimwal, Pandey and Pandey 2025) fits mcsGARCH to both Nifty 50 and Bank Nifty, so a referee will ask: does
the same hybrid, with nothing changed, also beat mcsGARCH on Bank Nifty? That is an out-of-sample test across
assets. It is worth something only if the design is fixed before any Bank Nifty outcome is seen, which is what
this file does.

## Prior knowledge (what was known when this was written)

* Every Nifty result of the v2 contest, including the numbers above. Section 9 uses its test statistics.
* Earlier studies in this repository used Bank Nifty data. None of them made an out-of-sample forecast:
  * `docs/EVENT_STUDY.md`: Bank Nifty realized variance around the expiry rule changes (for example, the last
    30 minutes of a Bank Nifty expiry day carried about 1.38 times the variance of the same half hour on other
    days).
  * `docs/REPLICATION_NSE.md`: mcsGARCH fitted in sample to two 64-day Bank Nifty windows (shock half-life 162
    and 38 minutes).
  * `docs/FALSIFICATION.md` T5: regressions of Bank Nifty's realized variance and end-of-day returns on GEX
    (for example F1a, GEX share, p = 0.0037; the F2 reversal, p = 0.0003). T5 also found that the
    last-30-minute returns of the two indices correlate 0.833 and that the sign of Pro GEX agrees on 98.2% of
    days.
  * `docs/CONNECTEDNESS_GAMMA.md`: an in-sample VAR of the minute volatility of both indices (shares of
    forecast-error variance, not a forecast evaluation).
  * `docs/PINNING_STUDY.md`, `docs/PINNING_MECHANISM.md` and `docs/INTRADAY_GAMMA.md`: pinning to strikes and
    intraday gamma on expiry days, with Bank Nifty included.

  None of these produced an out-of-sample forecast, a QLIKE or MSE loss, or any comparison of `hybrid_base` with
  mcsGARCH on Bank Nifty. Nothing in the hybrid (features, hyperparameters, settings) is changed here.
* In phase 1 only these Bank Nifty facts were looked at: which days have minute bars and how many; which days
  have a bhavcopy, an India VIX value and a participant file; expiry dates and their weekdays; chain sizes,
  strike spacing, days to expiry and lot sizes; futures rows; the number of missing values in the daily GEX and
  option-implied panels (built from option chains only); and, in the tests, the last bar's price LEVEL against
  NSE's official Bank Nifty close, to check that the right bars are read. Loading the market with the
  pipeline's own loader also fills an internal array of one-minute returns; no number made from it (variance,
  forecast, loss) was printed, saved or looked at.

## 1. The exact command

From the repository root, at the commit that adds this file, with the raw data listed in `data/MANIFEST.csv`:

```
OMP_NUM_THREADS=1 python scripts/run_pipeline.py --source nse --symbol BANKNIFTY --end 2026-04-22 \
    --n-seeds 5 --seed-study 3 --verbose --out results/pipeline_nse_banknifty \
    > results/pipeline_nse_banknifty_stdout.txt 2>&1
```

Every other option keeps its default, which is also the value the Nifty v2 run used
(`results/pipeline_nse_v2/run_info.json`): `--start 2023-01-01`, `--min-train 150`, `--block 21`,
`--mcs-refit-every 21`, `--seed 7`, `--targets next15 next30 last_hour`, `--seed-study-targets next30`,
`--oi-unit keep_units`, `--rate 0.065`, `--min-coverage 0.95`, `--holdout-start 2025-07-01`, no `--tune`, no
`--lean` and no `--lot-size`. The Nifty v2 run used the same options without `--symbol`, with its own `--out`,
and with no `--end` (its bars ended on 2 November 2026; `scripts/reproduce_all.py` now writes
`--end 2026-11-02`, which gives the same days). It took 99 minutes on one core for 792 test days; this run has
660.

`--end 2026-04-22` is the last day of the Bank Nifty bars (source sandeepkapri/BankNifty-Data, pinned at commit
6bdda6e571dd5e345ef3441df528a1fdccdbb0e9 in `data/raw/minute_bars_source.json`). It is written out so that a
later download with more days cannot change the sample.

Inputs (SHA-256 in `data/MANIFEST.csv`, checked against the files on 3 November 2026): `banknifty_1min.csv`
(83babf6f...), `india_vix.csv` (2c438728...), `index_daily.csv` (cec68df8...; its "Nifty Bank" rows are the
official trading days), and the same `bhavcopy/` and `participant_oi/` folders as the Nifty run (the Bank Nifty
rows of the bhavcopies).

## 2. Sample (checked in phase 1, calendar facts only)

* **Days:** 810 trading days, 2 January 2023 to 22 April 2026: every official "Nifty Bank" day with at least
  95% of the 375 one-minute bars of the 09:15 to 15:30 session. Two official days drop out, both Muhurat
  sessions: 1 November 2024 (60 bars, all in the evening, outside the session) and 21 October 2025 (60 bars).
  Six weekend days in the bars are not official days and are dropped, exactly as in the Nifty run: 12 November
  2023 (Muhurat), 20 January 2024, 2 March 2024, 18 May 2024, 1 February 2025 and 1 February 2026.
* All 810 days have a bhavcopy with Bank Nifty options (no chain is carried forward from an earlier day),
  Bank Nifty futures rows, an India VIX value for that date and a participant-wise open interest file.
* **Training only:** the first 150 days, 2 January to 9 August 2023.
* **Test days:** the 660 days from 10 August 2023 to 22 April 2026 (Nifty's first test day was also 10 August
  2023, and Nifty's test days up to 22 April 2026 are exactly these 660 dates), in 32 walk-forward blocks: 31
  of 21 days and a last one of 9. Before each block every model, mcsGARCH included, is refitted on all earlier
  days.
* **Forecasts:** next 15 and next 30 minutes from every 15th minute from 09:15 (25 and 24 forecasts a day);
  last hour: the 14:30 to 15:30 window only, with training on every 60-minute window that starts on a quarter
  hour (22 a day). LightGBM forecasts are the average of 5 seeds; the seed study reruns some LightGBM models
  with 3 single seeds at 30 minutes.
* **Test days by expiry regime** (the pipeline's labels): 312 up to 19 November 2024 (Bank Nifty had weekly
  options), 192 from 20 November 2024 to 31 August 2025 and 156 from 1 September 2025 (monthly options only).
* **Bank Nifty expiry days among the test days: 84**, of which 32 are monthly expiries (a Bank Nifty future
  expires that day).
  * 67 in the weekly period: 4 Thursdays in August 2023 (the last one the monthly); 53 Wednesdays from
    6 September 2023 to 13 November 2024; 6 last-Thursday monthlies from September 2023 to February 2024; and
    4 Tuesdays when a holiday moved a Wednesday expiry (16 April, 30 April, 16 July and 1 October 2024).
  * 17 in the monthly-only period: 10 from 27 November 2024 to 28 August 2025 and 7 from 30 September 2025 to
    30 March 2026.
* Nifty expires and Bank Nifty does not on 115 test days; both expire on 27.
* The daily GEX and option-implied panels (built from option chains only) were built in phase 1 to catch
  crashes, and only their missing values were counted. The only missing values are the gamma-flip level under
  the Pro sign rule (523 of 810 days, against 545 of 942 for Nifty: no flip within 4% of the forward), which the
  models treat as missing, as for Nifty.

## 3. What is identical, and every unavoidable difference

**Identical to the Nifty v2 run:** every model and its settings (`build_models` in `scripts/run_pipeline.py`),
the LightGBM hyperparameters and seeds, the feature lists (`BASE_FEATURES` and `GEX_SETS` in
`src/niftygex/features.py`), the training start and first test day, the walk-forward schedule (150 training
days, blocks of 21 days, every model refitted before each block), the losses, the Diebold-Mariano test, the
Holm adjustment, the Model Confidence Set and every option listed in Section 1. The code change (Section 11)
only chooses which files and rows are read (and which index counts as "the other index", D8) and changes
labels.

**Unavoidable differences:**

* **D1. Minute bars and sample length.** Bank Nifty's own third-party bars. In
  `results/data_validation/summary.csv` (811 days compared with NSE's official values) the day's high and low
  from the bars are within 5 basis points of NSE's on 99.4% and 99.3% of days (Nifty: 100% and 99.9%). The bars
  end on 22 April 2026, so there are 660 test days instead of 792 and **no closing-auction days** (NSE's
  closing auction starts on 3 August 2026). The `closing_auction` feature is 0 on every Bank Nifty day, and the
  pipeline's `pre_auction` and `auction_period` groups (and their Model Confidence Set files) do not exist.
  Nifty's "all days" included 63 auction days. A longer Bank Nifty sample would need another data source, which
  would add a second difference; we use what we have.
* **D2. Trading days.** The official list comes from the "Nifty Bank" rows of `index_daily.csv` instead of
  "Nifty 50". Over the shared dates the two lists give the same test days.
* **D3. Option chain.** GEX, the option-implied block and the `implied` and `mcsgarch_iv` models are built from
  Bank Nifty options in the same bhavcopies. Strikes are 100 points apart near the money (50 for Nifty) on all
  810 days. A day's nearest-expiry chain has 238 rows at the median (at least 142). Lot sizes differ (Bank Nifty
  25, 15, 30, 35, then 30 over the sample; Nifty 50, 25, 75, then 65; `docs/EVENT_STUDY.md` Table 2, and the
  files printed 15, 30, 35 and 30 when checked in phase 1) but do not enter: as in the Nifty run, open interest
  is kept in shares (`--oi-unit keep_units`), so GEX is in rupees per 1% move whatever the lot size.
* **D4. Expiry calendar.** Read from the bhavcopies (days on which Bank Nifty options really expired), with the
  monthly flag from Bank Nifty futures expiries. Bank Nifty had weekly options until 13 November 2024 (Thursday
  to 31 August 2023, Wednesday from 6 September 2023) and only monthly options from then on. So from 14
  November 2024 the "nearest expiry" contract, which feeds GEX and the near-expiry implied features, is a
  monthly up to 25 trading days away (for Nifty it was always a weekly, at most a week away). Its implied
  variance per day is then an average over the rest of the month, not the market's view of the next day
  (except on the day before an expiry). This changes what these features measure. It is part of the test and
  is not corrected.
* **D5. Futures depth.** The "GEX per unit of futures turnover" features use Bank Nifty futures traded value
  from the bhavcopies. In `daily_market.csv` the column is named `banknifty_futures_traded_value`.
* **D6. India VIX.** The `log_vix` feature and the `vix` model use India VIX, which is computed from Nifty
  options. NSE publishes no Bank Nifty volatility index and this repository has none. It is kept because
  replacing it would change the model; Bank Nifty's own implied volatility already enters through the implied
  block (D3). VIX should be a weaker signal for Bank Nifty than for Nifty.
* **D7. Participant-wise open interest.** The same NSE file, which covers all index options together, so the
  sign weights of the `participant` and `client` GEX sets are not specific to Bank Nifty (they were not specific
  to Nifty either). After November 2024 Bank Nifty is a smaller part of index options, so the weights describe
  its positioning even less well. This matters for H4 only.
* **D8. The "other index" feature `bn_expiry_only`.** For Nifty it flags days on which Bank Nifty expires and
  Nifty does not. For Bank Nifty it flags days on which Nifty expires and Bank Nifty does not (115 test days). The
  name is kept so that the feature list does not change. Reason: the feature means "the other big index expires
  today"; with Bank Nifty as its own other index it would be 0 on every day.
* **D9. Regime labels.** The pipeline's three regimes (cut at 20 November 2024 and 1 September 2025) were
  defined for Nifty, but the same two dates are Bank Nifty's own rule changes (weekly options gone; the monthly
  expiry moves to Tuesday), so they are kept. The names (`multi_weekly`, `single_weekly_thu`,
  `single_weekly_tue`) describe Nifty. Bank Nifty's other rule changes (weekly expiry from Thursday to Wednesday
  on 6 September 2023; monthly expiry from the last Thursday to the last Wednesday from March 2024; monthly back
  to the last Thursday from January 2025) fall inside regimes. The expiry flag, the monthly flag, days to expiry
  and the weekday follow them, because they come from the files.
* **D10. Development and hold-out.** For Nifty, test days before 1 July 2025 were the development window where
  the design was chosen. For Bank Nifty all 660 test days are out of sample with respect to the design. The
  pipeline still prints `development` (462 days) and `holdout` (198 days) groups; they are reported, carry no
  hypothesis and have no special meaning here.
* **D11. Labels.** Figure titles start with "BANKNIFTY:", the run label says "Bank Nifty bars", and the default
  output folder is `results/pipeline_nse_banknifty`. Cosmetic.

## 4. Hypotheses

Every test reads the pipeline's own output files, computed exactly as for Nifty. QLIKE = y/f - log(y/f) - 1 for
each forecast (y the realized variance, f the forecast), averaged within each day. Diebold-Mariano (DM) test on
the daily averages: Newey-West (HAC) variance, Harvey-Leybourne-Newbold correction, Student t with n - 1 degrees
of freedom, one-sided with the alternative that the second model has the lower loss. Holm adjustment:
`niftygex.evaluation.holm`.

**H1 (primary, fixed by the project lead before this file was written).** On all 660 test days, `hybrid_base`
has a lower mean QLIKE than `mcsgarch` for the next 15 minutes and for the next 30 minutes.
*Test:* the rows with subset `all`, loss `qlike` and comparison `mcsgarch vs hybrid_base` in
`tests_next15.csv` and `tests_next30.csv` (column `p`, the one-sided DM p); Holm over these two p-values.
*Success:* both Holm-adjusted p-values are at most 0.05. (With two tests this means: the smaller raw p is at
most 0.025 and the larger at most 0.05.)

**Secondary hypotheses.** Each has its own verdict at the level stated. They are not adjusted against each
other and cannot change the verdict on H1.

* **H2, last hour.** The same comparison for the 14:30 to 15:30 window on all test days: `tests_last_hour.csv`,
  subset `all`. One test; success if p is at most 0.05.
* **H3, Bank Nifty expiry days.** The same as H1 on the 84 expiry days: subset `expiry`, next 15 and next 30
  minutes, Holm over the two, success if both Holm-adjusted p are at most 0.05. The last hour on expiry days (84
  single windows) is reported without a hypothesis.
* **H4, GEX ablation: the prediction is no improvement.** For each GEX set (gross, US convention, participant
  (Pro), client mirror) and each of the three targets: the row with family `hybrid: base vs base+GEX`, that set's
  label and group `all` in `ablation_<target>.csv`. Column `p_one_sided` is the one-sided DM p that
  `hybrid_gex_<set>` has a lower loss than `hybrid_base`; `p_holm_in_family` is Holm over the four sets within
  one target, as for Nifty. *Prediction:* none of the 12 Holm-adjusted p-values (4 sets times 3 targets) is at
  most 0.05. If one or more is, the prediction has failed. Such a cell is then shown next to the placebo set
  (`hybrid_gex_placebo` against `hybrid_base`, and the "placebo vs real (same columns)" row), as was done for
  Nifty, but the verdict stays "prediction failed". The same table on expiry days is reported without a verdict.

## 5. What counts as replication

The verdict rests on H1 only:

* **Replicated:** both Holm-adjusted p-values are at most 0.05.
* **Partly replicated:** exactly one of the two is at most 0.05 (the write-up says which horizon).
* **Not replicated (failure):** neither is at most 0.05. Two cases get a name:
  * "same direction, not significant": `hybrid_base` has the lower mean QLIKE at both horizons;
  * "reversed": at either horizon `mcsgarch` has the lower mean QLIKE with a one-sided DM p of at most 0.05 in
    its favour (that p is 1 minus the `p` in the same row).

The size of the gain is compared with Nifty's only descriptively (Section 7, item 5). There is no test of the
difference between the two indices.

## 6. The expiry-regime change (weekly to monthly)

1. The calendar comes from the files (D4), so each day's expiry flag, monthly flag and days to expiry are
   correct in every regime.
2. The models are not changed: the regime code and calendar features work as for Nifty, and nothing is re-tuned
   for the monthly-only period.
3. H1, H2 and H4 pool all regimes, as for Nifty: 312 test days in the weekly period and 348 (192 + 156) in the
   monthly-only period.
4. Reported without a hypothesis: the H1 comparison in each of the three regimes (the pipeline's groups
   `multi_weekly`, `single_weekly_thu`, `single_weekly_tue`). The first regime is the weekly period (to 19
   November 2024; its last three days, 14, 18 and 19 November 2024, already had only a monthly Bank Nifty
   contract left). The other two together are the monthly-only period (from 20 November 2024), whose pooled
   comparison is computed from `forecasts_<target>.csv` with the same functions (`daily_losses`, then
   `diebold_mariano` with `alternative="greater"`).
5. 67 of the 84 expiry days are in the weekly period, so H3 mostly tests weekly expiries. The 17 monthly-only
   expiry days are reported, but they are too few to test.
6. mcsGARCH's expiry-day intraday curve is estimated from all expiry days in the training window (31 at the
   first refit, 98 at the first refit after 20 November 2024, 115 at the last). After November 2024 it is
   therefore dominated by weekly expiries. This is how the paper's model works and it is kept.

## 7. What will be reported, whatever the result

In `docs/BANKNIFTY_FORECAST.md`, in plain English, every number labelled REAL DATA:

1. The H1 verdict (Section 5) in the first paragraph, with both models' mean QLIKE, the change in percent, the
   DM statistic, the raw and Holm-adjusted p-values and the number of days.
2. H2, H3 and H4 with their verdicts.
3. A table like Section 4.1 of `docs/MODEL_ITERATIONS_REAL.md` for every model: all days and expiry days, three
   targets, QLIKE relative to mcsGARCH with the one-sided DM p.
4. The same breakdowns as for Nifty: by regime and by weekly against monthly-only period (Section 6), the 09:15
   forecast against later ones (`gain_by_anchor_<target>.csv`), trimmed QLIKE, MSE with its DM p-values, Model
   Confidence Set membership, the `development` and `holdout` groups (marked as having no special meaning here),
   and the seed noise.
5. **Nifty on the same dates:** the existing Nifty v2 forecasts (`results/pipeline_nse_v2/forecasts_<target>.csv`,
   not rerun) restricted to the same 660 test dates (10 August 2023 to 22 April 2026), scored the same way, next
   to the Bank Nifty numbers. This keeps Nifty's later days (including its 63 auction days) out of the
   comparison. The Nifty models on those dates were still trained with Nifty's own walk-forward, so this is a
   description, not a test.
6. The data checks of Section 2, recomputed by the run.
7. Every deviation (Section 8), any crash and how it was handled.

## 8. Deviations policy

* After this file is committed, the design does not change. Any change is a deviation and is listed in
  `docs/BANKNIFTY_FORECAST.md` with the date, the reason, and whether any Bank Nifty outcome had been seen when it
  was decided.
* The result of the command in Section 1 stays the headline. The result of a deviation is shown next to it, never
  instead of it.
* A crash is fixed with the smallest change that makes the run finish, and only if the fix changes no number on
  the days that already ran. A real bug that also affects Nifty is fixed for both indices: both are rerun and the
  old and new numbers of both are reported.
* After any Bank Nifty outcome has been seen, none of these may change for the headline: models, features,
  hyperparameters, seeds, sample dates or days, losses, tests, the subsets the hypotheses use, or the success
  thresholds.
* A data problem found in the Bank Nifty files (for example a bad day of bars) is reported. A sensitivity run
  without the affected days may be shown next to the headline, which keeps all days.
* If the run cannot be completed as specified, that is reported with the reason, and no verdict is given.

## 9. Power (from Nifty's published numbers only)

On Nifty the one-sided DM statistics for `mcsgarch vs hybrid_base` were 5.97 (next 15 minutes) and 7.94 (next 30
minutes) on 792 days (`results/pipeline_nse_v2/tests_<target>.csv`). If Bank Nifty's daily loss gap were as
strong relative to its noise, the expected statistics on 660 days would be about 5.5 and 7.2 (scaled by the
square root of 660/792), and H1 would succeed almost surely. At half that strength (about 2.7 and 3.6) H1 would
succeed with probability about 0.84; at a quarter (about 1.4 and 1.8), about 0.21. (Normal approximation,
treating the two tests as independent.) So H1 can detect a gain about half as strong as Nifty's, but probably
not a much weaker one. H3 (84 days) and H4 have far less power: for H4, "no improvement" means "no improvement
shown", as it did for Nifty.

## 10. Known weaknesses of this test

1. **It is not independent of Nifty.** Same dates, same news, the same VIX and participant file, and highly
   correlated returns (Prior knowledge). A success shows that the design is not specific to Nifty's data; it is
   not a second, independent sample in time.
2. **The project is not blind to Bank Nifty.** Earlier studies computed Bank Nifty volatility and GEX results (see
   Prior knowledge), although none made a forecast and none of them shaped the hybrid.
3. **Features change meaning after November 2024** (D4, D7, D8): the option features describe a monthly
   contract, and the participant weights fit Bank Nifty less well.
4. **mcsGARCH's expiry curve is a fixed shape** learnt mostly from weekly expiries (Section 6, item 6). If the
   hybrid's gain comes mainly from monthly expiry days after November 2024, part of it is the calendar features
   repairing that shape, not new information. The breakdowns in Section 7 show where the gain comes from.
5. **VIX is Nifty's** (D6).
6. **A shorter sample with no auction days** (D1), and third-party bars of slightly lower measured quality.

## 11. Code change, and how the Nifty path was checked

* `scripts/run_pipeline.py`: a `--symbol` option (`NIFTY`, the default, or `BANKNIFTY`) and a small table
  `SYMBOLS` that gives each index's bar file and its name in `index_daily.csv`. For `BANKNIFTY` it also changes
  the futures column name, the default output folder and the figure titles (D5, D11). `load_market` falls back to
  Nifty when it is called without a symbol, as `scripts/forward_test.py` does. `--symbol` is refused for the
  synthetic source.
* `src/niftygex/data.py`: `build_market_from_nse` gets `other_symbol`, the index whose expiry dates feed
  `bn_expiry_only` (D8): BANKNIFTY when the index is NIFTY (as before), NIFTY when it is BANKNIFTY.
  (`experiments/falsification.py` T5 builds a Bank Nifty market too but never reads these dates, so its outputs
  do not change.)
* `src/niftygex/features.py`: comments only.
* `tests/test_symbol_banknifty.py`: 11 tests. On SYNTHETIC files holding both indices, the Bank Nifty path reads
  Bank Nifty's chain (100-point strikes), lot size, futures and Wednesday expiry calendar, and `bn_expiry_only`
  marks Nifty's expiry days; the Nifty market is the same whether or not Bank Nifty rows are in the files. The
  pipeline's wiring for each symbol is checked. On REAL DATA (skipped when `data/raw` is absent): Bank Nifty's
  expiry dates from October 2024 to February 2025, its monthly flags, Nifty's Thursdays as the other index,
  official days only, the last bar within 1% of NSE's official Bank Nifty close, 100-point strikes, lot sizes 15,
  30, 35 and 30; and the rebuilt Nifty daily panels equal the committed `results/pipeline_nse_v2` files exactly
  (June to August 2024).
* **Checks run on 3 November 2026.**
  * *The Nifty path is unchanged.* The Nifty v2 command with the new code, for the last-hour target only
    (`--source nse --end 2026-11-02 --n-seeds 5 --seed-study 3 --targets last_hour`, 33 minutes on one core),
    wrote 28 files. Compared byte for byte with the committed `results/pipeline_nse_v2` files: 26 are
    identical (the three daily panels `daily_market.csv`, `daily_gex_panel.csv` and `daily_implied_panel.csv`,
    `forecasts_last_hour.csv`, `scoreboard_last_hour.csv`, `tests_last_hour.csv`, both ablation tables, all six
    Model Confidence Set files, the gain-by-anchor, importance, coefficient and hyperparameter tables,
    `mcsgarch_full_sample.csv` and all 7 figures). `mcsgarch_params.csv` equals the last-hour rows of the
    committed file exactly (the committed one also holds the other two targets). `run_info.json` differs only
    in the run's own options (`end`, `targets`, `out`, the new `symbol` entry) and its timings. The same command
    with the code before the change gave the same 27 files byte for byte.
  * *Tests.* `tests/test_symbol_banknifty.py`: 11 passed, none skipped. The full suite (`python -m pytest -q`):
    250 passed, exit code 0.
