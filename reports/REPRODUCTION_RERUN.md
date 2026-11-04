# Reproducibility rerun of the quick analysis steps (REAL DATA, run by an AI agent)

Run on 4 November 2026 by an AI agent (Agent A of the backtest round) in a COPY of the repository at commit
`971103c` (the copy's `src/` and `tests/` were reset to the committed versions, because the working tree
held an uncommitted change to `src/niftygex/gex.py` by another agent). The real checkout was not touched
except for this file. Each step was run as `python scripts/reproduce_all.py --only <step> --no-check` with
`OMP_NUM_THREADS=1` and a 300-second limit, four steps at a time on a 4-core machine that was also running
other agents' jobs (so the seconds below are slower than a quiet machine). Every file a step wrote under
`results/` was compared with the committed file: byte-identical, numeric-identical (the same numbers to
1e-9, only float formatting or run-time fields such as timestamps and hashes differ), or differs (with the
largest absolute difference and its column for csv files). Driver and per-step logs:
`/tmp/claude-0/-home-user/e79a6425-7016-5765-9dd9-652f4c629e5e/scratchpad/agentA/repro/`.

## The answer first

Everything that reads only the committed pipeline outputs reproduces exactly or to float precision:
forecast_diagnostics, real_day_level, momentum_gex, closing_auction_check, rekey_check, referee_forecast,
referee_forecast_gw, p107 (count, run, posthoc), p116_posthoc_diurnal_curves, the three demo days and
check_oi_units give 29 byte-identical and 35 numeric-identical files and no numeric difference above 6.8e-12.
p103_economic_value and f2_economics are numeric-identical as well (largest difference 2.5e-9, in a
fee-in-rupees column); their remaining text differences are missing-value formatting and input hashes.
Every step that reads the raw NSE files gives different numbers, and the reason is the data, not the code:
the manifest check in the copy reports that `index_daily.csv`, `india_vix.csv` and the two minute-bar
files differ from `data/MANIFEST.csv`, that the whole `data/raw/extra_2022/` folder (2,360 files for the
2022 paper replication) is missing, and that the bhavcopy, participant and index-close folders hold an
extra day (3 November 2026). The rerun panels of pinning_study and intraday_gamma hold five more days
than the committed ones (28, 29, 30 October, 2 and 3 November 2026), so the committed results of the
bhavcopy-based studies were made from a raw snapshot that ended on 27 October 2026. The missing 2022
folder explains the expiry calendar shrinking from 178 to 156 rows in replicate_paper, the
`FileNotFoundError` in pinning_mechanism, and probably the `RuntimeError` in expiry_event_study and
referee_expiry (the Bank Nifty monthly-expiry weekday change is now found twice). Four steps timed out at
300 s on the shared machine (replicate_paper, intraday_gamma, connectedness_gamma, power_real); what they
had written before the kill is compared but their verdict is TIMED OUT. Status: VERIFIED for the 18
steps that read committed results (the 16 above plus p103 and f2_economics); FAILED or NOT REPRODUCED for the raw-data steps until the raw data on
disk match the manifest and `extra_2022` is downloaded; NOT RUN for the 61 other steps of the 92 (pipelines,
long studies, sensex, nse_full, finnifty, processed-work steps).

## Manifest check in the copy (`python scripts/data_manifest.py check`)

| Finding | Files |
| --- | --- |
| different from the manifest | `index_daily.csv`, `india_vix.csv`, `minute_bars_source.json`, `nifty_1min.csv` (and `banknifty_1min.csv` per the coordinator; the minute-bar files carry 3 November) |
| missing | all of `extra_2022/` (2,360 files: the Oct to Dec 2022 bhavcopies, index closes and `index_daily.csv` of the paper replication) |
| extra, ignored | `bhavcopy/fo_20261103.csv.gz`, `index_close/ind_close_all_03112026.csv`, `participant_oi/fao_participant_oi_03112026.csv` |

## Step table

Seconds are wall-clock on the shared machine. "Files" is the number of files under `results/` the step
wrote (its stdout file included). Identical = byte-identical to the committed file; numeric-identical =
same numbers to 1e-9 with only formatting, timestamps, hashes or run-time fields differing.

| Step | Seconds | Files | Verdict | Largest difference (file: value, column) |
| --- | --- | --- | --- | --- |
| check_oi_units | 20 | 1 | 1 identical | |
| forecast_diagnostics | 5 | 4 | 1 identical, 3 numeric-identical | tails_and_last_hour_tests.csv: 1.1e-13, dm_two_sided_p |
| real_day_level | 7 | 5 | 2 identical, 3 numeric-identical | r3_gex_regressions.csv: 3.7e-12, t |
| momentum_gex | 8 | 4 | 3 identical, 1 numeric-identical | regressions.csv: 2.6e-13, se_hac |
| closing_auction_check | 8 | 6 | 1 identical, 5 numeric-identical | r2_tuesday_split.csv: 3.6e-12, ci95_high_pct |
| replicate_paper | 300 | 11 | TIMED OUT at 300 s; 5 files written before the kill differ | validation.csv: 50, official_days; expiry_calendar.csv 178 rows committed vs 156 rerun (extra_2022 missing); day_counts.csv: 7, nifty_expiry_days; validation_by_day.csv 838 vs 837 rows |
| pinning_study | 268 | 21 | 3 identical, 18 differ (data: 944 vs 939 panel days) | t4_close_proxy_check.csv: 15.6, p95_abs_err_pts; most others differ by the 5 extra days in n |
| intraday_gamma | 300 | 7 | TIMED OUT at 300 s; 5 files differ (data: 942 vs 937 days); 1 new uncommitted file (anchor_panel.csv.gz) | descriptives_by_time.csv: 543, median_gross_gex_crore_per_1pct |
| dealer_sign | 290 | 26 | 4 identical, 10 numeric-identical, 12 differ (data) | kalman_states.csv: 0.193, wald; oos_summary.csv: 0.0089, pct_vs_none; kalman_params.csv: 1e-4, lr_q_vs_0 |
| expiry_event_study | 46 | 0 | FAILED, exit 1: RuntimeError "expected one weekday change for BANKNIFTY monthly Thu, found 2" (expiry calendar from the current bhavcopy set) | |
| connectedness_gamma | 300 | 2 | TIMED OUT at 300 s; only its stdout file written (121 lines differ) | |
| power_real | 300 | 2 | TIMED OUT at 300 s; estimates.csv written before the kill is numeric-identical | estimates.csv: 1.2e-12, politis_white_block_resid |
| rekey_check | 4 | 1 | 1 identical | |
| pinning_mechanism | 154 | 0 | FAILED, exit 1: FileNotFoundError data/raw/extra_2022/index_daily.csv (folder not downloaded) | |
| f2_economics | 13 | 13 | 9 identical, 2 numeric-identical, 2 differ (run_info.json: input hashes and the on-disk replication fields; stdout 2 lines) | no numeric csv difference |
| referee_forecast | 159 | 13 | 3 identical, 9 numeric-identical, 1 differ (stdout file, 13 lines: run times) | all csv numeric-identical |
| referee_forecast_gw | 4 | 4 | 1 identical, 3 numeric-identical | |
| referee_f2 | 215 | 19 | 10 identical, 5 numeric-identical, 4 differ (participant files now include 3 November) | item4_other_rules.csv: 2e-4, p; item6_angle_bootstrap_draws.csv.gz binary differs |
| referee_f2_positions | 89 | 10 | 3 identical, 2 numeric-identical, 5 differ (run_info hashes, stdout) | item8_horse_race.csv: 1.0e-12, stat (text columns differ in 72 detail rows) |
| referee_expiry | 51 | 0 | FAILED, exit 1: same RuntimeError as expiry_event_study (it calls its define_events) | |
| p103_economic_value | 14 | 17 | 1 identical, 11 numeric-identical, 5 differ only in run-time fields and missing-value formatting (checked by hand) | fee_grid.csv: 2.5e-9, fee_rs_per_crore_year |
| p107_count | 4 | 2 | 1 identical, 1 numeric-identical | |
| p107_run | 29 | 14 | 4 identical, 10 numeric-identical | |
| p107_posthoc | 4 | 2 | 2 identical | |
| p113_connectedness_settings | 253 | 19 | 1 identical, 18 differ; CAVEAT: it read `results/replication_nse/expiry_calendar.csv`, which replicate_paper was rewriting at the same time with the 2022-less calendar, so this is partly an artefact of the parallel run | baseline_check.csv: 2,520, n_obs; by_dimension.csv and spec_fit.csv: a MAD15 that is finite in one file and missing in the other; near_ties.csv 6 vs 15 rows |
| p116_posthoc_diurnal_curves | 81 | 2 | 2 identical | |
| demo_pick | 2 | 0 | ran, wrote nothing under results/ | |
| demo_2025_12_23 | 4 | 3 | 3 identical (png included) | |
| demo_2024_06_04 | 2 | 2 | 2 identical | |
| demo_2025_04_08 | 2 | 2 | 2 identical | |
| number_trace | 3 | 3 | 1 identical, 2 differ (trace.csv text columns: it compares the documents with the rerun results folders, which now hold the changed files above) | no numeric difference |

Totals over the 31 steps: 68 identical files, 66 numeric-identical, 80 differ (of which 11 are stdout
files), 1 new uncommitted file. Steps skipped before running (not in the list above) and why: none of the
assigned 31 was skipped up front; the four timeouts and four failures are in the table.

## Caveats

* The copy's `results/` started as the committed results plus other agents' untracked folders
  (`results/a5_perturbation/`, `results/p120_joint_option_market/`), which no step here touched.
* Files were attributed to steps by modification time and by the output path each step declares in
  `scripts/reproduce_all.py`; two stdout files written while several steps ran are marked uncertain in
  the driver's json (`referee_f2_positions_stdout.txt` is counted under referee_f2_positions).
* Steps ran four at a time. Only one pair shares an input through `results/`: p113_connectedness_settings
  reads the expiry calendar that replicate_paper rewrites; its differences therefore need a sequential
  rerun before they are blamed on data or code.
* The machine was shared with other agents' jobs, so the 300-second limit was hit by steps that the
  documents time at a few minutes; the timeouts say nothing about reproducibility.
* The missing `extra_2022/` and the two differing daily files mean `data/raw` is not the manifest's
  snapshot. Until it is, the bhavcopy-based studies (replicate_paper, pinning_study, intraday_gamma,
  dealer_sign, expiry_event_study, referee_expiry, pinning_mechanism, p113) cannot be reproduced to the
  committed numbers; nothing here shows that their code is at fault.

## Fix: the expiry-schedule detector and an end date for the raw inputs (REAL DATA, AI agent)

Cause, confirmed on `data/raw`: the `RuntimeError` in expiry_event_study and referee_expiry is not caused
by the 3 November 2026 bhavcopy. The same error occurs with the bhavcopies cut at 2 November. It comes
from the missing `data/raw/extra_2022/` folder: without the October to December 2022 files the Bank Nifty
monthly series starts with 25 January 2023 (a Wednesday, 26 January was a holiday), 23 February
(Thursday) and 29 March 2023 (Wednesday, 30 March was a holiday). `detect_schedule_changes` in
`experiments/expiry_event_study.py` seeded the starting weekday with the mode of the first three expiries,
which is Wednesday, and then reported a false Wednesday-to-Thursday change on 27 July 2023 in addition
to the real one of 30 January 2025, so `define_events` found two. With the 2022 files (three October to
December Thursdays first) the seed is Thursday and nothing goes wrong, which is why the committed run
never hit it. The scripts also read every bhavcopy on disk and the full minute-bar files with no end
date, so a refreshed download could never reproduce a committed run exactly: a reproducibility defect
in itself.

Change (smallest that works), in `experiments/expiry_event_study.py`:

* `detect_schedule_changes`: the starting weekday is now the latest weekday among the expiries that
  come before the first run of `confirm` consecutive expiries on one weekday (a holiday moves an expiry
  to the previous trading day, so shifted expiries sit earlier in the week than the schedule). When the
  sample opens with a confirmed run, as the committed data do, this is the first weekday, exactly as
  before; with no confirmed run at all, the old mode rule applies. An isolated weekday was already a
  shift and still is (23 November 2026 for Bank Nifty, if Tuesday 24 November is a holiday).
* `build_data(raw, end)` and a new `--end` argument (default `2026-11-02`, the committed snapshot;
  `none` reads everything on disk) that cut `index_daily.csv`, the bhavcopy file list
  (`bhavcopy_files`) and the minute bars at that date. `experiments/referee_expiry.py` gets the same
  `--end` and passes it through. No other file changed.
* Test: `tests/test_expiry_calendar_holiday_shift.py` (6 SYNTHETIC cases: holiday-shifted opening, the
  committed-style opening, an isolated Monday before a Tuesday holiday, a confirmed change still found
  once, an early change to a later weekday, and the end-date file filter). With the existing
  `tests/test_event_study.py`: 19 passed.

Checks in the scratch copy, REAL DATA, `data/raw` as it is today (no `extra_2022`):

| Check | Result |
| --- | --- |
| detector on the bhavcopies to 2 November and to 3 November | 7 changes, identical to `results/event_study/schedule_changes_detected.csv`; `define_events` returns a, b, c, c2, d |
| `reproduce_all.py --only expiry_event_study` (default `--end 2026-11-02`) | finishes, exit 0 in 125 s; `schedule_changes_detected.csv`, `events.csv`, `event_b_paper_window_raw_vs_relative.csv` and `fig2_event_c_did.png` byte-identical to the committed files; the other 20 files differ because the 2022 quarter now comes from the Thursday rule instead of the bhavcopies (400 rule rows vs 380, 2 committed 2022 expiries absent, 0 kind or shift mismatches on the 722 common dates) and because `index_daily.csv` and `india_vix.csv` on disk differ from the manifest |
| `expiry_event_study.py --end none` (full data, 3 November included) | finishes, exit 0 in 137 s |
| `reproduce_all.py --only referee_expiry` (default `--end`) | finishes, exit 0 in 203 s |

So the two steps run again on today's download, the regime detection on the committed snapshot is
unchanged, and a full byte-level reproduction of `results/event_study/` still needs the `extra_2022`
folder and the manifest's `index_daily.csv` and `india_vix.csv`. Status: VERIFIED for the fix (crash
gone, detection identical), PARTIALLY VERIFIED for byte-level reproduction (data snapshot differs).
