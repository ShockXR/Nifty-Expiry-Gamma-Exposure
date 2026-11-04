# P104 pre-registration: does the Nifty forecasting hybrid beat mcsGARCH on a third index (Fin Nifty), with nothing changed?

**Status.** Written on 3 November 2026 (phase 1) by an AI agent (Claude) working for Shreyan Thanki. **This design
is committed to git before any Fin Nifty outcome of this study has been computed.** No Fin Nifty forecast, forecast
loss, test statistic or forecast target (the realized variance of a 15-minute, 30-minute or last-hour window) has
been computed. The commit that adds this file is the time stamp; phase 2 (the run) starts only after that commit
exists. Phase 2 writes `results/pipeline_nse_finnifty/` (every output of the pipeline),
`results/pipeline_nse_finnifty_stdout.txt` (the log), `results/pipeline_nse_finnifty/p104/` (the registered
summaries) and `docs/THIRD_INDEX_FORECAST.md` (the write-up). The code that phase 2 runs is in the same commit as
this file (Section 11). All results will be REAL DATA.

## In plain words

On Nifty, the anchor paper's intraday volatility model (mcsGARCH), corrected by a machine-learning model
(LightGBM), forecast the next 15 and 30 minutes of volatility better than mcsGARCH alone. The same model, with
nothing changed, also won on Bank Nifty (P91, `docs/BANKNIFTY_FORECAST.md`). Here we run exactly the same model a
third time, on **Fin Nifty** (NSE's Nifty Financial Services index, options symbol FINNIFTY), over 604 trading days
(10 August 2023 to 27 January 2026), and compare it with mcsGARCH in the same way and with the same rules as P91.

* Win at both 15 and 30 minutes, with the test still passing after allowing for two tests: **replicates**.
* At one of the two: **partly replicates**. At neither: **does not replicate**.
* Without changing the verdict we also look at the last hour, at Fin Nifty's expiry days, and at whether dealer
  gamma (GEX) helps. For GEX we predict, as for Nifty and Bank Nifty, that it does not.

Fin Nifty is not a clean third sample: about a third of it is the same big banks that make up Bank Nifty, so it moves
closely with Bank Nifty and Nifty. It has no volatility index of its own, its weekly options stopped in November
2024, and its minute bars come from a third-party file of unknown origin that we checked against NSE's own daily
values. All of this is written down here before any Fin Nifty forecast is made.

## Feasibility: which third index, and why Fin Nifty (the data search)

The plan needed a third index with (a) free 1-minute index bars from January 2023 (or close) that pass the same
check as the Nifty and Bank Nifty bars (`scripts/validate_minute_bars.py`: the day's high and low from the bars
within 0.05% (5 basis points) of NSE's official high and low on nearly all days), (b) its own option chains in
NSE's bhavcopies and (c) enough days for the walk-forward. **Floor set before the search: at least about 400 test
days** (P91 had 660). Candidates and what happened (searched on 3 November 2026 with `gh api search/repositories`
and `gh api search/code`; files read from raw.githubusercontent.com at pinned commits):

| Candidate source | What it holds | Outcome |
| --- | --- | --- |
| BSE Sensex (`data/raw/sensex_1min.csv`, already in the repository) | Sensex 1-minute bars 2023 to 2026 | **Not eligible.** Sensex options trade on BSE, and BSE's servers cannot be reached from this machine, so Sensex's own option chain (needed by the implied and GEX features) cannot be downloaded. "Nothing changed" is impossible. |
| `deep776-max/FYERS_NSE_FINNIFTY_INDEX` (commit c404a45) | FYERS broker dumps of Fin Nifty index and option bars | Rejected: index bars only from 21 June 2026 to 27 October 2026 (about 90 days). |
| `deep776-max/FYERS_NSE_MIDCPNIFTY_INDEX` (commit 1579ef4) | the same for Midcap Select | Rejected: index bars only from 19 July 2026 to 27 October 2026. |
| `laughingbud/capstone` | intraday files of Fin Nifty and Midcap Select **futures**, 2022 and 2023 | Rejected: futures, not the index, and nothing after 2023. |
| `sreekanthmp/options_trading_model` | Angel One daily files and 5- and 15-minute files for one week of June 2026 | Rejected: not 1-minute, far too short. |
| `QuantDev-stack/OptionVault` (commit b2889a6) | a 50-row Fin Nifty 1-minute sample (6 January 2026) | Too short to use; used as an independent spot check (below). |
| `DoRmAmMu1997/My-Algo-Trading-Code` | a script that downloads Fin Nifty 1-minute bars from a broker account | No data files in the repository; needs an account. |
| Code search for Midcap Select 1-minute bars (`MIDCPNIFTY 1min`, `MIDCPNIFTY 09:15:00 extension:csv`, file name searches) | broker code, futures files, option chains | **No index minute bars for 2023 to 2026 found. Midcap Select rejected.** |
| **`ayush-roy-21/EigenTrade` (commit 21fc3e72cb6b54f55af2ef9815de6a7c140ac0f8)** | 17 files `data/FINNIFTY_part1.csv` to `part17.csv`: Fin Nifty 1-minute bars, 4 August 2021 to 27 January 2026 | **Passes; used.** |

Other repositories that the searches returned (trading apps, option-chain viewers, daily-data projects) hold no
1-minute index history and are not listed one by one. Kaggle and other sites that need a login were not tried.

**Check of the Fin Nifty bars (REAL DATA, levels only).** `python scripts/validate_minute_bars.py --indices "Nifty
Financial Services" --out results/data_validation_finnifty` compares each day's bars with NSE's official daily
values of "Nifty Financial Services" (from the daily index files already downloaded for P96). On the 755 days from 2
January 2023 to 27 January 2026: the day's high is within 5 basis points on **98.9%** of days and the low on
**99.2%**; the worst day is 13.4 basis points off; the median miss is 0.0 for both. Of the 14 days with a miss above
5 basis points, 6 lack the 09:15 bar (374 bars). Every official day has bars except the Muhurat session of 1
November 2024 (evening only). For comparison, P91 accepted Bank Nifty at 99.4% and 99.3% (worst 23.3 basis points),
and Nifty's bars score 100% and 99.9%. We call a source clean if at least 98% of days pass on the high and on the
low; this number is written here after the check was run, and Fin Nifty's bars are the weakest of the three, so the
reader can judge. **Independent spot check:** the 50 one-minute bars of 6 January 2026 (09:15 to 10:04) in the
OptionVault sample match the EigenTrade bars exactly (open, high, low and close, all 50). **Provenance caveat:** the
EigenTrade repository states no licence and does not say where the bars came from (dates are written in an Excel
style, `="DD-MM-YY"`; a volume column is zero before 2025 and filled on some later days). The raw files are kept out
of git, like the other third-party bars, and pinned by commit and SHA-256 (`data/raw/finnifty_1min_source.json`,
`data/MANIFEST.csv`).

**Verdict of the search: feasible with Fin Nifty**, 604 test days (above the 400 floor). Midcap Select and Sensex are
not feasible.

## Prior knowledge (what was known when this was written)

* Every Nifty result of the v2 contest and every Bank Nifty result of P91. On Bank Nifty the hybrid was 3.9% and
  5.0% better than mcsGARCH at 15 and 30 minutes (Holm p 5e-12 and 2e-11), 8.1% in the last hour; H3 passed; the
  H4 prediction failed at 15 minutes (GEX helped, all of it at the 09:15 forecast); `hybridiv` was the best model.
* Earlier studies used Fin Nifty data, none of them an intraday forecast:
  * `docs/SYNTHETIC_CONTROL.md` (P96): a daily high-low volatility measure of Fin Nifty against a synthetic control,
    before and after its weekly options ended (Fin Nifty's Tuesday premium moved by +0.28 log points, placebo p
    0.75; the study was far too imprecise to tell).
  * `docs/REFEREE_RESPONSE_EXPIRY.md`: Fin Nifty's Tuesday expiry calendar from the bhavcopies, and its expiry-day
    option turnover (0.72 to 1.11 of Nifty's on its Tuesday, 2022 to November 2024).
  * `docs/PINNING_MECHANISM.md` (C4): Fin Nifty expiry dates as a control.

  None of these computed a Fin Nifty 1-minute return, intraday realized variance, forecast or forecast loss.
* In phase 1 only these Fin Nifty facts were looked at: the daily level check above (high, low, open and close of
  the bars against NSE's official values); which days have bars and how many; which days have a bhavcopy, an India
  VIX value and a participant file; expiry dates and their weekdays; chain sizes, strike spacing, days to expiry
  and lot sizes; futures rows; and the number of missing values in the daily GEX and option-implied panels (built
  from option chains only). Loading the market with the pipeline's own loader also fills an internal array of
  one-minute returns; no number made from it (variance, forecast, loss) was printed, saved or looked at.

## 1. The exact command

From the repository root, at the commit that adds this file, with the raw data listed in `data/MANIFEST.csv`
(Fin Nifty files fetched with `python scripts/fetch_finnifty_data.py --raw data/raw`):

```
OMP_NUM_THREADS=1 python scripts/run_pipeline.py --source nse --symbol FINNIFTY --end 2026-01-27 \
    --n-seeds 5 --seed-study 3 --verbose --out results/pipeline_nse_finnifty \
    > results/pipeline_nse_finnifty_stdout.txt 2>&1
python experiments/p104_finnifty_summary.py
```

Every other option keeps its default, which is also the value of the Nifty v2 and Bank Nifty P91 runs:
`--start 2023-01-01`, `--min-train 150`, `--block 21`, `--mcs-refit-every 21`, `--seed 7`, `--targets next15 next30
last_hour`, `--seed-study-targets next30`, `--oi-unit keep_units`, `--rate 0.065`, `--min-coverage 0.95`,
`--holdout-start 2025-07-01`, no `--tune`, no `--lean`, no `--lot-size`. `--end 2026-01-27` is the last day of the
Fin Nifty bars; it is written out so that a later download cannot change the sample. If the run is done with
`PYTHONPATH` pointing at this commit's `src` (as P91 did, because the package is installed in editable mode from
another checkout), that changes no number.

Inputs (SHA-256 in `data/MANIFEST.csv`): `finnifty_1min.csv` (1ab71128...), `finnifty/index_daily_finnifty.csv`
(db240ce7...; the official trading days), the 764 files of `finnifty/bhavcopy/`, and the same `india_vix.csv` and
`participant_oi/` as the Nifty and Bank Nifty runs.

## 2. Sample (checked in phase 1, calendar facts only)

* **Days:** 754 trading days, 2 January 2023 to 27 January 2026: every official "Nifty Financial Services" day with
  at least 95% of the 375 one-minute bars of the 09:15 to 15:30 session. Two official days drop out, both Muhurat
  sessions (1 November 2024 and 21 October 2025, 60 bars each). Five weekend days in the bars are not official
  days and are dropped, exactly as in the Nifty run: 12 November 2023 (Muhurat), 20 January 2024, 2 March 2024,
  18 May 2024 and 1 February 2025.
* All 754 days have a bhavcopy with Fin Nifty options (no chain is carried forward), Fin Nifty futures rows, an
  India VIX value for that date and a participant-wise open interest file.
* **Training only:** the first 150 days, 2 January to 9 August 2023.
* **Test days:** the 604 days from 10 August 2023 to 27 January 2026, which are exactly Nifty's test days up to that
  date, in 29 walk-forward blocks (28 of 21 days and a last one of 16). Before each block every model, mcsGARCH
  included, is refitted on all earlier days.
* **Forecasts:** as for Nifty and Bank Nifty (25 a day at 15 minutes, 24 at 30 minutes, one last-hour window at
  14:30 with training on every 60-minute window that starts on a quarter hour; LightGBM forecasts are the average
  of 5 seeds; the seed study reruns some LightGBM models with 3 single seeds at 30 minutes).
* **Test days by regime** (the pipeline's labels): 312 up to 19 November 2024 (Fin Nifty had weekly options), 192
  from 20 November 2024 to 31 August 2025, and 100 from 1 September 2025 (monthly options only).
* **Fin Nifty expiry days among the test days: 82**, of which 30 are monthly expiries (a Fin Nifty future expires
  that day).
  * 67 in the weekly period: Tuesdays, and 4 Mondays when a holiday moved the Tuesday expiry (14 August,
    18 September, 23 October and 13 November 2023). The last weekly expiry was 19 November 2024.
  * 15 in the monthly-only period: last Tuesday (26 November and 31 December 2024), last Thursday (8, January to
    August 2025) and last Tuesday again (5, September 2025 to 27 January 2026). The last test day, 27 January 2026,
    is an expiry day.
* Nifty expires and Fin Nifty does not on 117 test days; both expire on 13.
* No closing-auction days (the auction starts on 3 August 2026), so the `closing_auction` feature is 0 on every
  day and the `pre_auction` and `auction_period` groups do not exist, as for Bank Nifty.
* **Missing values in the daily panels** (built in phase 1 to catch crashes; only missing values counted). GEX
  panel: the Pro-rule gamma-flip level on 468 of 754 days (Bank Nifty 523 of 810, Nifty 545 of 942: no flip within
  4% of the forward), the US-convention flip level on 5 days, and the at-the-money implied volatility of the chain
  on 1 day. Implied panel: the next-expiry columns (`iv_next`, `n_next`, `w_next`, `fwdvar_day`) on 11 days, which
  makes the features `log_fwdvar_day` and `exp_premium` missing on those days. Unlike Bank Nifty, these few gaps
  exist; the models treat them exactly as on Nifty (LightGBM handles missing values itself; the linear models fill
  them with the training median).

## 3. What is identical, and every unavoidable difference

**Identical to the Nifty v2 and Bank Nifty P91 runs:** every model and its settings (`build_models` in
`scripts/run_pipeline.py`), the LightGBM hyperparameters and seeds, the feature lists (`BASE_FEATURES` and `GEX_SETS`
in `src/niftygex/features.py`), the training start and first test day, the walk-forward schedule, the losses, the
Diebold-Mariano test, the Holm adjustment, the Model Confidence Set and every option in Section 1. The code change
(Section 11) only chooses which files are read and changes labels.

**Unavoidable differences** (D1 to D11 follow P91's list; D12 and D13 are new):

* **D1. Minute bars and sample length.** Fin Nifty's own third-party bars (feasibility section): high and low within
  5 basis points of NSE's on 98.9% and 99.2% of days, the weakest of the three sources, of unknown origin and with no
  stated licence. The bars end on 27 January 2026: 604 test days (Bank Nifty 660, Nifty 792), no auction days.
* **D2. Trading days.** The official list comes from the "Nifty Financial Services" rows of NSE's daily index files
  (`nse_full/index_close_full/`, downloaded for P96), written to `finnifty/index_daily_finnifty.csv`, because
  `index_daily.csv` holds only Nifty 50, Nifty Bank and India VIX. Over the sample this gives the same test days as
  Nifty's list.
* **D3. Option chain and its folder.** GEX, the option-implied block and the `implied` and `mcsgarch_iv` models are
  built from Fin Nifty options. The main `bhavcopy/` folder was filtered to Nifty and Bank Nifty rows when it was
  downloaded, so Fin Nifty uses its own folder `finnifty/bhavcopy/` with every index option and future of each day
  in NSE's own columns: 512 files are byte copies of P96's `nse_full/fo_index/` (1 August 2023 to 29 August 2025) and
  252 were downloaded from NSE's archive on 3 November 2026 with the same URLs as `scripts/download_nse.py` (2 January
  to 31 July 2023 and 1 September 2025 to 6 February 2026; the loader reads files up to 10 days past the last bar day,
  as it did for Bank Nifty). The Nifty rows of the copied files equal those of `bhavcopy/` (tested). Strikes near the
  money are 50 points apart on 739 of 754 days and 100 points on 15. A day's nearest-expiry chain has 192 rows at the
  median (at least 110). Lot sizes (40, 25, 65, then 60 in the files) do not enter, because open interest is kept in
  shares (`--oi-unit keep_units`).
* **D4. Expiry calendar.** Read from the bhavcopies, with the monthly flag from Fin Nifty futures expiries. Weekly
  options (Tuesday) to 19 November 2024, then monthly only (Section 2). From 20 November 2024 the "nearest expiry"
  contract that feeds GEX and the near-expiry implied features is a monthly up to 25 trading days away, so those
  features describe the rest of the month, not the next day. As for Bank Nifty, this is part of the test and is not
  corrected.
* **D5. Futures depth.** The "GEX per unit of futures turnover" features use Fin Nifty futures traded value; the
  column in `daily_market.csv` is `finnifty_futures_traded_value`.
* **D6. India VIX.** NSE publishes no Fin Nifty volatility index. `log_vix` and the `vix` model use India VIX (built
  from Nifty options), as for Bank Nifty. Fin Nifty's own implied volatility enters through the implied block.
* **D7. Participant-wise open interest.** The same NSE file, covering all index options together. Fin Nifty is a
  small part of index options, so the sign weights of the `participant` and `client` GEX sets describe its
  positioning even less well than Bank Nifty's. This matters for H4 only.
* **D8. The "other index" feature `bn_expiry_only`.** For Fin Nifty it flags days on which **Nifty** expires and Fin
  Nifty does not (117 test days). Fin Nifty has two big neighbours; we take Nifty because it is by far the largest
  index option market, because P91 used Nifty as Bank Nifty's other index, and because the code's rule is then one
  simple rule ("Bank Nifty for Nifty, Nifty for every other index"). Bank Nifty would also have been defensible
  (about a third of Fin Nifty is Bank Nifty's banks); this choice is made now, before any outcome. The name is kept
  so that the feature list does not change.
* **D9. Regime labels.** The pipeline's cuts at 20 November 2024 and 1 September 2025 are again Fin Nifty's own rule
  changes (weekly options gone after 19 November 2024; monthly expiry back to Tuesday from September 2025), so they
  are kept. The move of the monthly expiry from Tuesday to Thursday in January 2025 falls inside the second regime;
  the expiry, monthly, days-to-expiry and weekday features follow it because they come from the files.
* **D10. Development and hold-out.** All 604 test days are out of sample for the design. The pipeline still prints
  `development` (462 days) and `holdout` (142 days) groups; they carry no hypothesis.
* **D11. Labels.** Figure titles start with "FINNIFTY:", the run label says "Fin Nifty bars", the default output
  folder is `results/pipeline_nse_finnifty`. Cosmetic.
* **D12. The sample is a sub-period of Bank Nifty's.** The 604 test dates are a subset of P91's 660 and of Nifty's,
  so the three results share dates, news, VIX and the participant file.
* **D13. Fewer missing-value-free panels** (Section 2): 11 days without next-expiry implied values, 5 without a
  US-convention flip and 1 without a chain at-the-money volatility, handled as on Nifty.

## 4. Hypotheses

Exactly as in P91 (Section 4 there), on Fin Nifty's outputs. QLIKE = y/f - log(y/f) - 1 per forecast, averaged
within each day. One-sided Diebold-Mariano test on the daily averages (Newey-West variance, Harvey-Leybourne-Newbold
correction, Student t with n - 1 degrees of freedom), alternative: the second model has the lower loss. Holm:
`niftygex.evaluation.holm`.

**H1 (primary).** On all 604 test days, `hybrid_base` has a lower mean QLIKE than `mcsgarch` for the next 15 minutes
and for the next 30 minutes. *Test:* rows `all`, `qlike`, `mcsgarch vs hybrid_base` of `tests_next15.csv` and
`tests_next30.csv` (column `p`); Holm over the two. *Success:* both Holm-adjusted p-values at most 0.05.

**Secondary hypotheses** (each with its own verdict; not adjusted against each other; cannot change H1):

* **H2, last hour.** The same comparison for the 14:30 to 15:30 window, all days (`tests_last_hour.csv`, subset
  `all`). Success if p is at most 0.05.
* **H3, Fin Nifty expiry days.** As H1 on the 82 expiry days (subset `expiry`), Holm over the two horizons, success
  if both Holm-adjusted p are at most 0.05. The last hour on expiry days is reported without a hypothesis.
* **H4, GEX ablation, predicted no improvement.** For each GEX set (gross, US convention, participant (Pro), client
  mirror) and each target: family `hybrid: base vs base+GEX`, group `all` in `ablation_<target>.csv`, column
  `p_holm_in_family` (Holm over the four sets within one target). *Prediction:* none of the 12 Holm-adjusted p-values
  is at most 0.05. If one is, the prediction has failed, and the cell is shown next to the placebo set as in P91. The
  table on expiry days is reported without a verdict. (P91's Bank Nifty prediction failed at 15 minutes; we keep the
  same prediction here because "nothing changed" applies to the hypotheses too.)

## 5. What counts as replication

As in P91: **replicated** if both H1 Holm-adjusted p-values are at most 0.05; **partly replicated** if exactly one is
(the write-up names the horizon); **not replicated** if neither, with the named cases "same direction, not
significant" (`hybrid_base` has the lower mean QLIKE at both horizons) and "reversed" (at either horizon `mcsgarch`
has the lower mean QLIKE with a one-sided p of at most 0.05 in its favour, that is 1 minus the row's `p`). The size of
the gain is compared with Nifty's and Bank Nifty's only descriptively.

## 6. The expiry-regime change (weekly to monthly)

As in P91 Section 6: the calendar comes from the files; nothing is re-tuned for the monthly-only period; H1, H2 and
H4 pool all regimes (312 test days in the weekly period, 292 in the monthly-only period); the H1 comparison by regime
and the pooled weekly against monthly-only split (computed from `forecasts_<target>.csv` with `daily_losses` and
`diebold_mariano(..., alternative="greater")`, cut at 20 November 2024) are reported without a hypothesis; 67 of the
82 expiry days are in the weekly period, so H3 mostly tests weekly expiries; mcsGARCH's expiry-day intraday curve is
estimated from all expiry days in its training window, as the paper's model does.

## 7. What will be reported, whatever the result

In `docs/THIRD_INDEX_FORECAST.md`, in plain English, every number labelled REAL DATA:

1. The H1 verdict in the first paragraph, with both models' mean QLIKE, the change in percent, the DM statistic,
   the raw and Holm-adjusted p-values and the number of days.
2. H2, H3 and H4 with their verdicts.
3. A table of every model against mcsGARCH (all days and expiry days, three targets, QLIKE change and one-sided p).
4. The same breakdowns as P91: by regime and weekly against monthly-only, 09:15 against later anchors, trimmed QLIKE,
   MSE, Model Confidence Set membership, development and hold-out groups (no special meaning), seed noise.
5. **Nifty and Bank Nifty on the same dates:** the committed Nifty v2 forecasts and the committed Bank Nifty P91
   forecasts (neither rerun), restricted to the 604 Fin Nifty test dates and scored the same way (each index with
   its own expiry days). A description, not a test.
6. The data checks of Section 2, recomputed by the run.
7. Every deviation, any crash and how it was handled.

## 8. Deviations policy

The same as P91 Section 8: after this commit the design does not change; any change is a deviation listed in the
write-up with its date, its reason and whether a Fin Nifty outcome had been seen; the command of Section 1 stays the
headline and any deviation is shown next to it; a crash is fixed with the smallest change that changes no number on
days already run (a real bug that also affects Nifty or Bank Nifty is fixed for all and all are rerun); a data problem
found in the Fin Nifty files is reported, and a sensitivity run without the affected days may be shown next to the
headline, which keeps all days; if the run cannot be completed as specified, no verdict is given.

## 9. Power (from the published Nifty and Bank Nifty numbers only)

On Nifty the H1 DM statistics were 5.97 and 7.94 on 792 days; on Bank Nifty 7.03 and 6.72 on 660 days. Scaled to 604
days (square root of the ratio of days), a Fin Nifty gap as strong relative to its noise would give about 5.2 and 6.9
(Nifty-like) or 6.7 and 6.4 (Bank Nifty-like), and H1 would succeed almost surely. At half that strength, H1 succeeds
with probability about 0.80 (Nifty-like) or 0.90 (Bank Nifty-like); at a quarter, about 0.18 or 0.23 (normal
approximation, the two tests treated as independent, Holm rule as in Section 5). So H1 can detect a gain about half
as strong as on the first two indices, but probably not a much weaker one. H3 (82 days) and H4 have much less power;
for H4 "no improvement" means "no improvement shown".

## 10. Known weaknesses of this test

1. **Not independent of Nifty or Bank Nifty** (D12): same dates, same news, the same VIX and participant file, and Fin
   Nifty shares many constituents with Bank Nifty. A success shows the design is not specific to one index's data; it
   is not a new sample in time.
2. **The bars are the weakest of the three and of unknown origin** (D1). They pass the level check on nearly all days
   and match an independent 50-minute sample exactly, but nothing checks every minute.
3. **The project is not blind to Fin Nifty** (Prior knowledge), and P91's Bank Nifty results were known when this plan
   was written, though nothing in the model was changed in response.
4. **Features change meaning after November 2024** (D4, D7, D8), as for Bank Nifty.
5. **mcsGARCH's expiry curve is one fixed shape** learnt mostly from weekly expiries; the breakdowns show where the
   gain comes from.
6. **VIX is Nifty's** (D6). **A shorter sample with no auction days** (D1).

## 11. Code change, and how the other paths were checked

* `scripts/run_pipeline.py`: `FINNIFTY` added to the `SYMBOLS` table (bar file `finnifty_1min.csv`, official name
  "nifty financial services", label "Fin Nifty") with two new optional keys, `bhavcopy` (the folder of F&O
  bhavcopies, default `bhavcopy`) and `official_file` (the file with the official days, default `index_daily.csv`),
  which `load_market` now reads. For NIFTY and BANKNIFTY both keys keep their defaults, so the files read are the
  same as before.
* `src/niftygex/data.py`: the default "other index" rule is now "BANKNIFTY for NIFTY, NIFTY for every other index"
  (before: "NIFTY for BANKNIFTY, BANKNIFTY for every other index"). For NIFTY and BANKNIFTY nothing changes.
* `src/niftygex/features.py`: comments only.
* `scripts/fetch_finnifty_data.py` (new): downloads and pins the Fin Nifty data (bars with SHA-256 checks, the
  official-day file, the bhavcopy folder).
* `scripts/validate_minute_bars.py`: a `--indices` option so that the Fin Nifty bars can be checked against their
  own official file; the default output (`results/data_validation/`) is byte for byte unchanged (checked).
* `experiments/p104_finnifty_summary.py` (new): a copy of `experiments/p91_banknifty_summary.py` with the paths
  changed and Bank Nifty's P91 forecasts added to the same-dates comparison. It fits nothing.
* `tests/test_symbol_finnifty.py`: 8 tests. On SYNTHETIC files holding three indices, the Fin Nifty path reads Fin
  Nifty's chain, lot size, futures and Tuesday expiry calendar, and `bn_expiry_only` marks Nifty's expiry days; the
  Nifty and Bank Nifty markets are the same with or without Fin Nifty rows, and their other index is unchanged. The
  pipeline's wiring for each symbol is checked. On REAL DATA (skipped when the files are absent): Fin Nifty's expiry
  dates from October 2024 to February 2025, its monthly flags, Nifty's Thursdays as the other index, official days
  only, the last bar within 1% of NSE's official Fin Nifty close, strike spacing, futures, and the copied bhavcopies'
  Nifty rows equal to those of `bhavcopy/`.
* **Checks run on 3 November 2026.** `tests/test_symbol_finnifty.py`: 8 passed. `tests/test_symbol_banknifty.py`
  with the new code: 11 passed (it includes the test that the rebuilt Nifty daily panels equal the committed
  `results/pipeline_nse_v2` files). The full suite (`python -m pytest -q`): 520 passed, exit code 0. P91's 33-minute
  byte-for-byte rerun of Nifty's last hour was not repeated: the only code paths that changed for NIFTY and
  BANKNIFTY are the two dictionary lookups with unchanged defaults and the rewritten default rule, which the tests
  cover.
