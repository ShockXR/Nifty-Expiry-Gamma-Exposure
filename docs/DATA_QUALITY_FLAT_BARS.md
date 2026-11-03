# Data quality: flat minutes in the one-minute index files (DQ1)

**REAL DATA.** Post hoc check, made on 3 November 2026 after the second AI referee report asked for "the flat-bar screen as a declared data-quality rule for all minute-bar studies". It was run by a separate AI agent, read only, and then rerun inside the repository with the scripts in `experiments/dq1_flat_bars/`; all outputs are in `results/dq1_flat_bars/`. Nothing in it is a registered test, and nothing was added to the test ledger.

**Short answer.** Two NSE outage days (10 July 2017 and 24 February 2021) and a few non-regular sessions show long runs of an unchanged price. None of them is inside a forecasting sample, a replication window or any study from 2023 on. The only study with one inside its windows is P117, which already reports it. No headline conclusion changes. The rule below (DQ1) is declared for every new minute-bar study from now on.

## 1. What was measured

A **flat minute** is a one-minute return of exactly zero: the index closes a minute at exactly the same price as the
minute before. A few of these are normal. Long stretches of them mean the data feed stopped and the file kept
repeating the last price, as on 24 February 2021, when an NSE outage halted trading.

How it was counted (`screen.py`): the project's own loader (`niftygex.data.load_minute_bars` and
`to_session_grid`, the forecasting pipeline's code), the 09:15 to 15:30 session (375 one-minute slots), and the
pipeline's return definition (each minute against the previous minute's close; the 09:15 minute against its own
open). The **flat share** is flat minutes divided by the minutes that have a return. For each day it also records
the longest run of consecutive flat minutes and its clock times, and compares the day's highest and lowest bar with
NSE's official daily high and low (the logic of `scripts/validate_minute_bars.py`, 5 basis point tolerance).

Files screened (`coverage.csv`): Nifty 50, 2,368 days (3 Apr 2017 to 2 Nov 2026); Bank Nifty, 2,297 days (2 Jan 2017
to 22 Apr 2026); Fin Nifty, 988 days (4 Aug 2021 to 27 Jan 2026); Sensex, 948 days (2 Jan 2023 to 2 Nov 2026).

**Official values exist locally only from 20 Oct 2022** (Nifty, Bank Nifty: `data/raw/extra_2022/index_daily.csv`
and `data/raw/index_daily.csv`) **and 2 Jan 2023** (Fin Nifty: `data/raw/finnifty/index_daily_finnifty.csv`). There
is no BSE file for Sensex. So none of the days flagged at 20% can be checked against NSE's high and low here.

## 2. Flagged days

### More than 20% flat minutes (`flat_days_over_20pct.csv`)

| Index | Day | Bars | Flat minutes | Flat share | Longest flat run | What the day is |
| --- | --- | --- | --- | --- | --- | --- |
| Nifty | Mon 10 Jul 2017 | 375 | 167 | 44.5% | 60 min (10:00 to 10:59) | NSE trading outage day |
| Nifty | Wed 24 Feb 2021 | 375 | 303 | 80.8% | 238 min (10:07 to 14:04) | NSE outage; the file repeats 14814.5, then moves from 14:05, then is flat again at the close |
| Nifty | Thu 4 Nov 2021 | 43 | 42 | 100% | 42 min (14:48 to 15:29) | Diwali special session; only 43 bars fall in the regular session window |
| Bank Nifty | Mon 10 Jul 2017 | 375 | 181 | 48.3% | 110 min (09:25 to 11:14) | NSE trading outage day |
| Fin Nifty | Sat 7 Aug 2021 | 312 | 148 | 48.2% | 66 min | Saturday, not a regular trading day |
| Fin Nifty | Sat 21 Aug 2021 | 56 | 55 | 100% | 55 min | Saturday |
| Fin Nifty | Thu 4 Nov 2021 | 43 | 42 | 100% | 42 min | Diwali special session |
| Fin Nifty | Sat 13 Nov 2021 | 374 | 142 | 38.1% | 112 min | Saturday |
| Fin Nifty | Sat 8 Jan 2022 | 68 | 68 | 100% | 68 min | Saturday |
| Fin Nifty | Sat 22 Jan 2022 | 29 | 29 | 100% | 29 min | Saturday |
| Sensex | none | | | | | |

Official high/low check: not possible for any of these days (all before the official files start).

The event study's own measure (bars whose high equals their low) flags exactly the same ten days at 20%
(`flat_all_days.csv.gz`, column `high_eq_low_bars`).

### Between 5% and 20% flat minutes (`flat_days_over_5pct.csv`)

| Index | Day | Flat minutes | Flat share | Longest run | Note |
| --- | --- | --- | --- | --- | --- |
| Nifty | Wed 20 Sep 2017 | 20 | 5.3% | 1 min | scattered single flat minutes, no stall |
| Nifty | Fri 13 Mar 2020 | 51 | 13.6% | 44 min (09:22 to 10:05) | market-wide circuit-breaker halt (a real halt, not a feed fault) |
| Nifty | Mon 23 Mar 2020 | 50 | 13.3% | 43 min (09:59 to 10:41) | market-wide circuit-breaker halt |
| Nifty | Mon 7 Mar 2022 | 40 | 10.8% | 25 min (09:57 to 10:21) | unexplained stall; Bank Nifty has only 325 bars that day |
| Nifty | Wed 19 Aug 2026 | 19 | 5.1% | 14 min (15:15 to 15:28) | closing-auction freeze; bars' high and low equal NSE's (0.0 bp) |
| Bank Nifty | Fri 13 Mar 2020 | 58 | 15.5% | 51 min | circuit-breaker halt |
| Bank Nifty | Mon 23 Mar 2020 | 57 | 15.2% | 50 min | circuit-breaker halt |
| Bank Nifty | Mon 7 Mar 2022 | 26 | 8.0% | 11 min | 325 bars only |

### From 2023 onward (the forecasting samples), from `flat_all_days.csv.gz`

* No day of any index is above 5% except Nifty 19 Aug 2026.
* Before the closing auction started (3 Aug 2026), the longest flat run on any day is 2 minutes (Nifty, Fin Nifty)
  or 1 minute (Bank Nifty, Sensex).
* From 3 Aug 2026 every Nifty and Sensex day has 13 to 19 flat minutes (3.5% to 5.1%), almost all from 15:15 to
  15:28: this is the auction freeze the project already treats as a separate regime, not a data fault.
* The 30 days on which bars and NSE's official high or low differ by more than 5 basis points have at most 5 flat
  minutes each. So the existing high/low validation and the flat screen catch different problems; neither replaces
  the other.

## 3. Which studies contain flagged days

A study is affected only if a flagged day is inside its sample and it uses minute returns from that day.

| Study | Minute-data sample | 20% days inside | Uses their minute returns? | Verdict |
| --- | --- | --- | --- | --- |
| Forecasting pipeline, Nifty (`results/pipeline_nse_v2`) | 2 Jan 2023 to 2 Nov 2026 | none | | not affected (19 Aug 2026 at 5% is an auction day, already split out) |
| Bank Nifty and Fin Nifty pipelines | from 2 Jan 2023 | none | | not affected; same for P103, P106, P107, P109, P110, P116, P118, seed bootstrap |
| Replication (`results/replication_nse`) | windows 14 Nov 2022 to 10 Feb 2023, 1 Nov 2023 to 29 Feb 2024, 1 Mar to 31 May 2024 | none in windows | only daily closes of 10 Jul 2017 in the daily EGARCH history | not affected (recheck 4.2) |
| P113 connectedness settings | windows from 14 Nov 2022 | none | no | not affected |
| P114 (`results/p114_persistence_fixed_model`) | windows from 14 Nov 2022; rolling windows 14 Nov 2022 onward | none in any window | yes, indirectly: model F's daily level is a regression on daily realized variance from every grid day since 2017 (10 Jul 2017; plus 24 Feb 2021 on the one-index rolling grid) | technically affected; recheck 4.2 shows no material change |
| P117 (`results/p117_persistence_placebo`) | windows slide over 2018 to 2026 | 24 Feb 2021 inside 123 Nifty pairs; 10 Jul 2017 in history | yes | affected (already reported post hoc in P117); recheck 4.1 |
| Event study (`results/event_study`) | from 4 Mar 2019 | 24 Feb 2021, 4 Nov 2021 | no: already dropped by its own 20% rule (`results/event_study_stdout.txt`, lines 20 and 21) | not affected at 20%; 5% days kept, recheck 4.3 |
| Referee expiry checks, Sensex confound | rebuild the event study's days | as above | no | not affected |
| Pinning study | Nifty closes from 2023, expiry days only | none | | not affected |
| Falsification and F2 reversal | daily data from 2023 | none | | not affected (19 Aug 2026 is in F2's separate auction sample) |
| Frozen live forecast (`results/live/2026-11-03/forecast.json`) | bars from 1 Dec 2022, 920 training days | none | | not affected |
| Intraday gamma, connectedness gamma, Sensex swap, synthetic control, dealer sign | from Oct 2022 or later | none | | not affected |

All Fin Nifty flagged days are Saturdays or the Diwali session before 2023. They never enter a study: the pipelines
start in 2023, and P117's Fin Nifty control uses P114's one-index grid, which drops weekend days and days under
300 bars.

## 4. Cheap rechecks (post hoc, descriptive, nothing added to the test ledger)

### 4.1 P117, the only study with a flagged day inside its windows

Placebo p without the pairs that contain a flagged day (`recheck_p117.csv`; the 20% rows reproduce the committed
`results/p117_persistence_placebo/posthoc_without_flat_days.csv` exactly):

| Nifty | Registered p | Without >20% days | Without >5% days |
| --- | --- | --- | --- |
| ref, primary set | 0.191 | 0.154 | 0.198 |
| F, primary set | 0.105 | 0.081 | 0.104 |
| ref, clean set (no verdict registered) | 0.095 | 0.008 | 0.012 |
| F, clean set (no verdict registered) | 0.057 | 0.002 | 0.002 |

Bank Nifty: no pair contains a 20% day, so nothing changes; without the 5% days p moves from 0.113 to 0.143 (ref)
and 0.096 to 0.105 (F) in the primary set.

Single windows refitted without 24 Feb 2021 (`recheck_p117_windows.csv`, same code as P117; the committed half-lives
are reproduced exactly): with the day, the reference model's half-life sits at the persistence bound (about 67,000
to 69,000 minutes) in all four windows tried; without it, 187 to 430 minutes. Model F: from about 69,000 minutes
(three windows) or 178 minutes (one window) to 27 to 78 minutes. One broken day decides these fits.

### 4.2 P114 and the replication's Table 2 (history only)

Refit on the actual windows with the flagged days removed from the history (`recheck_p114_fits.csv`,
`recheck_p114_fall.csv`; the registered rows reproduce `results/p114_persistence_fixed_model/fits.csv` exactly).
Half-life in minutes, before (14 Nov 2022 to 10 Feb 2023) and after (1 Mar to 31 May 2024):

| Model, index | Registered | Without >20% days | Without >5% days |
| --- | --- | --- | --- |
| F, Nifty | 141.05 to 17.91 (change in equivalent persistence -0.0331) | 141.07 to 17.91 (-0.0331) | 141.64 to 17.95 (-0.0330) |
| F, Bank Nifty | 62.66 to 10.47 | 62.76 to 10.49 | 63.71 to 10.49 |
| ref (Table 2), Nifty | 358.64 to 72.44 | 358.48 to 72.39 | 359.36 to 72.09 |
| ref (Table 2), Bank Nifty | 161.97 to 37.96 | 161.48 to 37.91 | 165.23 to 37.95 |

Changes are under 2.1% in every half-life. P114's registered verdicts and the replication's Table 2 stand.

### 4.3 Event study, days between 5% and 20% that it keeps

Rebuilding the weekly values without Nifty 13 Mar 2020, 23 Mar 2020, 7 Mar 2022 and Bank Nifty 13 and 23 Mar 2020,
with the study's own code and seeds (`recheck_event_study.csv`; committed values reproduced exactly). Regime R0
(Mar 2019 to Sep 2023), own expiry day, weeks without monthly expiries:

| Measure | Committed | Without the 5% days |
| --- | --- | --- |
| Nifty, 5-minute RV (primary) | -0.163 (HAC p 0.0001) | -0.156 (p 0.0002) |
| Bank Nifty, 5-minute RV | -0.106 (p 0.010) | -0.100 (p 0.015) |
| Bank Nifty, last 30 minutes | +0.106 (p 0.040) | +0.109 (p 0.035) |
| Bank Nifty, 1-minute RV (frequency diagnostic) | -0.078 (p 0.048) | -0.072 (p 0.071) |

Only the last row crosses 0.05, and it is a diagnostic, not a headline. The 2023 to 2025 event tests never reach
these days.

### 4.4 The two outage days against NSE's official values

The local official files start in October 2022, so `experiments/dq1_flat_bars/official_check.py` downloads NSE's daily index file for each outage day from the public archive (`archives.nseindia.com/content/indices/ind_close_all_DDMMYYYY.csv`, saved under `data/raw/dq1_official/`, not committed) and compares it with the 09:15 to 15:29 bars, as `scripts/validate_minute_bars.py` does (`official_check.csv`):

| Index, day | Bars | High: bars / official (bp) | Low: bars / official (bp) | Last close / official close (bp) |
| --- | --- | --- | --- | --- |
| Nifty, 10 Jul 2017 | 375 | 9,782.15 / 9,782.15 (0.0) | 9,646.60 / 9,646.45 (+0.2) | 9,763.10 / 9,771.05 (-8.1) |
| Nifty, 24 Feb 2021 | 375 | 14,836.40 / 15,008.80 (-114.9) | 14,681.20 / 14,723.05 (-28.4) | 14,781.30 / 14,982.00 (-134.0) |
| Bank Nifty, 10 Jul 2017 | 375 | 23,960.30 / 23,964.80 (-1.9) | 23,449.15 / 23,289.90 (+68.4) | 23,644.45 / 23,675.05 (-12.9) |
| Bank Nifty, 24 Feb 2021 | 54 | 35,637.20 / 36,567.65 (-254.4) | 35,106.90 / 35,060.65 (+13.2) | 35,626.60 / 36,452.30 (-226.5) |

On 24 February 2021 the official high and close of both indices lie far above every bar in the files, so the files are not a usable record of that session (the official values are consistent with trading after 15:30 that day, which the files do not cover; we did not check the session times against an NSE circular). On 10 July 2017 Nifty's bars match the official range within 0.2 basis points, so its flat stretch is consistent with a real pause in trading rather than a broken file, while Bank Nifty's bars miss the official low by 68 basis points. Three of the four index-days are above 20% flat, so DQ1 removes them; Bank Nifty on 24 February 2021 has only 54 bars and is already removed by the existing bar-count minimum.

## 5. Could a headline conclusion change?

No. No flagged day lies in the forecasting samples, the replication windows, the pinning, F2 or live samples. The
event study already applies the 20% rule. P114 and Table 2 move by under 2.1%. P117's registered primary verdict
("within the normal range") holds with or without the flagged pairs (Nifty p 0.154 and 0.081 at 20%). What does
change is P117's clean-set reading, which had no registered verdict: without 24 Feb 2021 almost no clean placebo pair
falls as far as the actual pair (p 0.008 and 0.002). P117 already reports this as post hoc.

## 6. Recommended declared rule

**DQ1, flat minutes.** On the minute grid a study estimates on (09:15 to 15:30 session, or its declared trim), a day
whose share of exactly zero one-minute returns is above 20% is removed before any minute-level estimation: window
fits, daily realized measures, and the history that daily models are fitted on. Each study lists the days it
removed.

* Days between 5% and 20% stay in but are listed. If one falls inside an estimation window, the study reports a
  run without it.
* The closing-auction freeze from 3 Aug 2026 (13 to 19 flat minutes, at 15:15 to 15:28) is handled by the existing
  auction split, not by DQ1.
* The existing rules stay: official trading days only, the bar-count minimum, and the 5 basis point high/low
  validation.
* Past studies are not rerun. They cite this note as their sensitivity, and P117 keeps its post hoc tables.
* The two outage days were checked against NSE's official daily values (Section 4.4): 24 February 2021 cannot be reconciled with them for either index.

## 7. Files and commands

Scripts in `experiments/dq1_flat_bars/`, outputs in `results/dq1_flat_bars/`: `screen.py` writes `coverage.csv`, `flat_all_days.csv.gz`, `flat_days_over_20pct.csv` and `flat_days_over_5pct.csv` (the screen); `recheck_p117.py` writes `recheck_p117.csv` (P117 p-values); `recheck_p117_window.py` writes `recheck_p117_windows.csv` (window refits); `recheck_p114.py` writes `recheck_p114_fits.csv` and `recheck_p114_fall.csv` (P114 and Table 2); `recheck_event_study.py` writes `recheck_event_study.csv` (event study); `official_check.py` writes `official_check.csv`. Each `*_stdout.txt` holds the printed output. Run them in that order from the repository root, with data/raw in place:

```bash
for s in screen recheck_p117 recheck_p117_window recheck_p114 recheck_event_study official_check; do
  PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/dq1_flat_bars/$s.py
done
```

The rerun inside the repository reproduced the AI agent's first outputs byte for byte (`coverage.csv`, both flagged-day lists, `flat_all_days.csv.gz` after decompression, and every recheck file; `recheck_p114_fall.csv` agrees in every value, with one column, the change in equivalent persistence, now computed in the script rather than added by hand).
