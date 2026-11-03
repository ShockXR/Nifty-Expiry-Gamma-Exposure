# Backtest and stress report (Agent 4): niftygex

Scope: independent recomputation of the headline forecast result from the committed files, a look-ahead probe on the real code path, a stress test of the GEX engine, and a data-integrity check. Neither prototype is a trading bot, so there is no Sharpe ratio here: the metrics are forecast loss (QLIKE), Diebold-Mariano tests, and failure classes.

## Summary in plain language

1. The headline numbers are right. I rebuilt the loss for every model from the raw forecast and realized columns with my own code (no niftygex import) and got the committed scoreboard to the last digit. The hybrid is 5.80% and 7.65% better than our mcsGARCH at 15 and 30 minutes, 4.43% and 6.64% better than the paper-as-printed model, and my day-level Diebold-Mariano statistics (6.409, 8.340, 3.801 against the paper-as-printed model) equal the committed ones. [VERIFIED]
2. The 15 and 30 minute gain is robust: it is positive in every half, every calendar year, every one of 11 rolling half-year windows, and after dropping the biggest-loss observations or the closing-auction period.
3. The last-hour gain (6.4%) is weaker than it looks: the hybrid wins on only 53.5% of days (sign test p = 0.0506), the 2025 calendar year shows no gain at all (-0.01%), and 2 of 11 rolling windows are negative. [PARTIALLY VERIFIED]
4. I found one real defect in the code: the choice between the daily EGARCH model and its EWMA fallback uses data from after the refit date, so forecasts for early test days change when only later data change. This is a small look-ahead in the benchmark that every learned model builds on. Its effect on the committed headline could not be measured because the raw data are not in the repository. [FAILED probe, impact UNVERIFIED]
5. The GEX engine does the arithmetic right but does not check its inputs: a chain that failed to download, an expired chain, or a zero implied volatility all produce a net GEX of exactly 0.0 (a "neutral day") with no error. Unit mistakes (shares for contracts) are not detected either. 24 of 41 stress cases were silently accepted.

## Phase status

| Phase | What | Status |
| --- | --- | --- |
| A1 | Recompute QLIKE, gains and DM tests from committed forecasts; outliers, trimming, subperiods, closing auction | VERIFIED (headline), PARTIALLY VERIFIED (last hour) |
| A2 | Look-ahead probe, placebo, negative control on the real code path with SYNTHETIC data | FAILED (look-ahead found in daily EGARCH selection); placebo VERIFIED |
| A3 | GEX engine stress (41 cases) | VERIFIED (cases run); engine has silent failures |
| A4 | Real-data integrity of the minute bars | PARTIALLY VERIFIED (bars byte-identical to the manifest; 69-day sample against NSE agrees; other files not fetched) |
| A5 | Rerun of the hybrid with perturbed seeds, learning rate and feature noise | BLOCKED: needs the raw NSE data (about 25 minutes of downloads) and a pipeline rerun; not attempted after the container restart |

Labels: REAL = committed forecasts from the real NSE-based run; SYNTHETIC = niftygex.simulate output; OUR CHECK = computed here.

## A1. Recomputation from the committed forecasts [REAL data, OUR CHECK code]

Files: `results/pipeline_nse_v2/forecasts_{next15,next30,last_hour}.csv` (19,800 / 19,008 / 792 rows, 792 test days, 2023-08-10 to 2026-11-02) and `results/p116_benchmark_as_printed/daily_losses_nifty_*.csv` (day-level losses of the paper-as-printed model, mcs_cpp, whose raw forecasts are not committed).
Loss: QLIKE = y/f - log(y/f) - 1. Test: day-level loss differences, Newey-West variance (Bartlett, lag floor(4 (n/100)^(2/9))), Harvey-Leybourne-Newbold factor, Student t(n-1), as in the repo, re-implemented independently.

### Baseline

| Check | next15 | next30 | last hour | Status |
| --- | --- | --- | --- | --- |
| Max difference of my mean QLIKE from the committed scoreboard, all 32 models | 0 | 0 | 0 | VERIFIED |
| Hybrid gain over our mcsGARCH (mcsgarch column), % | 5.80 | 7.65 | 6.36 | VERIFIED (quoted 5.8, 7.6) |
| Hybrid gain over paper-as-printed model (mcs_cpp), % | 4.43 | 6.64 | 6.16 | VERIFIED (quoted 4.4, 6.6; last hour as in P116) |
| DM statistic vs mcs_cpp, mine / committed | 6.409 / 6.4088 | 8.340 / 8.3396 | 3.801 / 3.8007 | VERIFIED |
| DM statistic vs our mcsGARCH (NW lag 5 to 21 give the same conclusion) | 5.97 | 7.94 | 3.73 | OUR CHECK |
| Stationary-bootstrap 95% interval of gain vs our mcsGARCH, % | 4.54 to 7.12 | 6.30 to 9.06 | 3.40 to 10.03 | OUR CHECK (seed 9,600,001, 5,000 draws, mean block 10 days) |
| Bootstrap 95% interval of gain vs mcs_cpp, % | 3.41 to 5.40 | 5.41 to 7.92 | 3.31 to 10.06 | OUR CHECK |
| Share of days the hybrid beats our mcsGARCH | 71.3% | 74.1% | 53.5% | OUR CHECK |
| Sign test p (two-sided) | 5e-34 | 2e-43 | 0.0506 | OUR CHECK |
| Daily file of P116 equals my daily loss of the hybrid and of mcs_ours | yes (1e-15) | yes | yes | VERIFIED |

### Simple baselines (gain of hybrid_base over each, %, one-sided DM p)

| Baseline | next15 | next30 | last hour |
| --- | --- | --- | --- |
| naive_last | 69.0 (8e-24) | 52.8 (2e-34) | 52.2 (6e-8) |
| HAR | 11.3 (4e-14) | 14.4 (7e-11) | 8.1 (0.004) |
| VIX | 38.8 (2e-8) | 44.3 (3e-7) | 22.3 (0.021) |
| Option-implied | 33.5 (1e-26) | 37.6 (2e-21) | 25.8 (5e-4) |
| mcsgarch_iv (sees option prices) | 4.8 (0.006) | 5.9 (6e-4) | 5.2 (0.016) |
| lgbm_base (the learner alone, no mcsGARCH offset) | 2.3 (0.16) | 4.3 (0.18) | 0.2 (0.48) |
| linear_base | 3.7 (0.016) | 3.9 (0.006) | -0.6 (0.62) |

Reading: the hybrid beats every simple baseline clearly, but it is not significantly better than the learner alone (lgbm_base) at any horizon. The hybrid's advantage over mcsGARCH therefore comes largely from the learner, which fits the repo's own statement that option-market information is not the source.

### Integrity checks on the committed forecast files [VERIFIED]

No duplicate (date, minute) rows; dates and the day index t nondecreasing; anchor minutes increase within each day; the largest minute + horizon is 375 (the session length), so no window runs past the close; no NaN, zero or negative forecast in any of the 32 model columns; no NaN or non-positive realized value. Whether a forecast uses only information before its window cannot be seen in the csv; that is what A2 probes.

### Outliers, trimming, subperiods, closing auction (gain of hybrid over our mcsGARCH, %)

| Scheme | next15 | next30 | last hour |
| --- | --- | --- | --- |
| All observations | 5.80 | 7.65 | 6.36 |
| Drop only 5 June 2024 09:15 | 5.80 | 7.65 | 6.36 |
| Drop top 1% by joint loss | 5.52 | 6.40 | 6.60 |
| Drop top 5% by joint loss | 4.23 | 5.00 | 4.14 |
| Drop top 5% by mcsGARCH's own loss | 3.37 | 3.66 | 3.41 |
| Each model trimmed at its own top 5% | 4.68 | 5.47 | 4.73 |
| Drop top 5% by absolute loss difference (the most hostile rule) | 2.38 | 2.77 | 3.21 |
| First half / second half of test days | 5.78 / 5.82 | 6.93 / 8.38 | 7.95 / 5.63 |
| Development (before 2025-07-01) / hold-out | 5.34 / 6.38 | 6.61 / 9.24 | 5.27 / 7.10 |
| Calendar 2023 / 2024 / 2025 / 2026 | 5.4 / 5.8 / 4.1 / 7.6 | 5.1 / 7.2 / 6.2 / 10.7 | 8.3 / 9.3 / -0.01 / 7.7 |
| 126-day rolling windows (step 63): min, max, positive count | 2.80, 8.78, 11 of 11 | 4.96, 9.88, 11 of 11 | -3.26, 12.20, 9 of 11 |
| Exclude windows that touch the frozen index (from 3 Aug 2026) | 5.71 | 7.20 | 7.01 |
| Only windows that touch the frozen index | 6.80 | 16.61 | 5.25 |
| Expiry days / non-expiry days | 6.64 / 5.50 | 6.23 / 8.13 | 7.91 / 5.95 |
| Exclude the 09:15 anchor | 5.08 | 5.96 | not applicable |

Findings from the table:
* The 5 June 2024 09:15 forecast (mcsGARCH about 20 times too high the morning after the election crash) is invisible under QLIKE (loss 2.06 against typical 0.28) but it is 79% of mcsGARCH's total squared error at 15 minutes and 76% of the hybrid's. Under MSE the hybrid is 4.45 times worse than HAR with that one observation and 1.08 times without it. The repo already says this; I confirm it. [VERIFIED]
* The observations that dominate QLIKE are not that one: they are the first closing-auction days (3 and 4 August 2026, minutes 345 and 360, losses 188 and 82) and the election day itself (4 June 2024, 09:15, loss 76). The headline survives their removal.
* The gain falls under the most hostile trimming (about 2.4% to 3.2%) but stays positive.
* The last-hour gain is carried by a minority of days and has no gain in 2025. It should be quoted with its interval (3.4% to 10.0%) and the 53.5% win share.

Status: headline at 15 and 30 minutes VERIFIED; last-hour claim PARTIALLY VERIFIED.

## A2. Look-ahead probe on the real code path [SYNTHETIC data, OUR CHECK]

Setup: `niftygex.simulate.simulate_market` (300 days, kappa 0, seed 9,700,001), `build_gex_panel`, `build_dataset` (horizon 30, step 45), `run_walkforward` (min_train 100, block 21, models naive_last, har, mcsgarch, linear_base, lgbm_base, hybrid_base). On this world the hybrid gains 4.86% over mcsGARCH (DM p 3e-6), so there is a gain that a probe can test.

| Probe | Result | Status |
| --- | --- | --- |
| Replace all days from 170 on by an independent world (seed 9,700,002); compare the 75 numeric feature columns for days before 170 | identical | VERIFIED |
| Same replacement; compare forecasts for days before 170 (560 rows, the block 163 to 183 straddles the cut) | NOT identical: first differing day is 100, the first test day; mcsgarch differs by up to 2.4 times, hybrid_base by up to 1.7 times; naive_last is identical | FAILED |
| Positive control: forecasts after the cut change | yes | VERIFIED |
| Cause | `daily_egarch_variance` chooses between EGARCH and an EWMA fallback with `_egarch_ok(params, h, y_train)`, where `h` is the filtered variance over the WHOLE series including future days; the `fix` filter itself returns different `h[:n]` for the same fitted parameters when the future changes (checked: parameters equal, `h[:n]` not equal). With the future changed, refit 100 switches between "ewma" and "egarch_normal". `_ewma_variance` also floors at 1% of the full-sample mean square. | root cause: implementation defect |
| My probe-only patch (check on the filter of the training window only, floor from the training window) | does NOT remove the difference (mcsgarch up to 5.7 times): the full-series `h` from the arch library also depends on the future | the fix needs a filter run on data up to the end of each block only |
| Real-data scale of the problem (OUR CHECK, daily return approximated by log(close_last_bar/open) from `daily_market.csv`, so only indicative) | the method chosen with the full series differs from the choice with data up to refit + 21 days at 4 of 38 refits (starts 171, 192, 213, 234); the committed run itself used the EWMA fallback at refits 171 and 192 | PARTIALLY VERIFIED |
| Effect of the defect on the committed hybrid-versus-mcsGARCH gain | cannot be measured without the raw minute data | UNVERIFIED |
| Label-shuffle placebo B2: training labels permuted within each anchor minute (10 shuffles, seeds 9,710,000 to 9,710,009), scored on the REAL labels | hybrid gain over mcsGARCH falls from +4.86% to -19.45% (every shuffle between -14.9% and -24.9%) | VERIFIED |
| Label-shuffle design B1 (train and test labels both shuffled) | hybrid gain +27% to +34%: this is NOT a leak, it shows that a recalibrating model beats a fixed structural model on unrelated labels, so this design cannot be used as a placebo for a gain over mcsGARCH | evaluation-design caution |
| Negative control: days permuted (5 permutations, seeds 9,720,000 to 9,720,004) | the hybrid still gains 2.0% to 13.9% (mean 6.3%) over mcsGARCH, so the gain over mcsGARCH on synthetic data does not by itself prove it comes from time-ordering information | invalid assumption if read as proof of information content |

Why the repo's own test does not catch it: `tests/test_forecast_v2.py` removes the last days and compares days before a block boundary; the defect needs a refit block that straddles the cut and a changed (not removed) future.

## A3. GEX engine stress (41 cases, SYNTHETIC chains, seed 9,800,000)

Classes: DETECTED 3, SAFELY HANDLED 11, SILENTLY IGNORED 24, CRASHED 3. Full table: `reports/stress/a3_gex_stress_cases.csv`.

| Case | Result | Class |
| --- | --- | --- |
| Clean chain (control) | net GEX -4.0e10, flip 23,651 | control |
| Open interest all NaN / IV all NaN / T = 0 / T < 0 / IV = 0 / spot = 0 / empty chain | net GEX exactly 0.0 (pandas sum skips NaN); only a RuntimeWarning in some | SILENTLY IGNORED |
| 20% of OI NaN, 25% of IV NaN | rows silently dropped, net GEX -7.5e9 and -1.6e11 instead of -4.0e10 | SILENTLY IGNORED |
| Negative OI, negative IV, negative lot size, weights outside [-1, 1] | accepted, sign or scale wrong | SILENTLY IGNORED |
| OI in shares instead of contracts | exactly 75 times larger, no unit check | SILENTLY IGNORED |
| Lot size 1 instead of 75 | 1.3% of correct | SILENTLY IGNORED |
| Option type as strings "CE"/"PE" | every string casts to True, all options become calls, net = gross | SILENTLY IGNORED |
| Duplicated rows near the money; calls only; no strikes within 2% of spot | net GEX changes by factors or flips sign, no coverage check | SILENTLY IGNORED |
| Zero OI everywhere | net 0 (correct) but call wall, put wall and max pain return the first strike | SILENTLY IGNORED |
| T of one minute or one second | finite, scales as 1/sqrt(T) (2.2e13 at one second); callers must floor T | SAFELY HANDLED |
| Wide strike range, deep OTM only, 1/0 integer option type, strings as strikes | correct | SAFELY HANDLED |
| IV solver round trip (20,000 random options, 1 hour to 30 days) | every OTM option solved (0 NaN of 4,541 with price above 0.05); NaN only for ITM options with no resolvable time value; relative price error below 1e-11 | SAFELY HANDLED |
| Impossible option prices to the IV solver | NaN | DETECTED |
| Unknown sign rule; participant rule without weights | ValueError with a clear message | DETECTED |
| Missing column (iv, oi), max_pain on an empty chain | KeyError / numpy ValueError, loud but not informative | CRASHED |

Root causes: implementation gaps (no input validation, `Series.sum()` with default skipna), invalid assumptions (units), data problems (coverage). Not tested: the pipeline-level guards that may run before the engine (for example the coverage threshold and the bhavcopy-missing flag), so how many of these reach the committed results is UNVERIFIED.

## A4. Real-data integrity of the minute bars

Fetched with the repo's own `scripts/fetch_minute_bars.py --out <scratch> --until 2026-11-02` (about 10 seconds; the Nifty 50 and Bank Nifty bars come from public GitHub mirrors) into a scratch folder, not into `data/raw`. Scripts: `reports/stress/a4_minute_bar_integrity.py`, output `a4_minute_bar_integrity.csv`; the repo's own `validate_minute_bars.py` output for the sample window is `a4_validate_minute_bars_summary_2024-05_to_2024-08.csv`.

| Check | Result | Committed value | Status |
| --- | --- | --- | --- |
| SHA-256 of `nifty_1min.csv` (46,934,128 bytes) and `banknifty_1min.csv` (46,473,079 bytes) against `data/MANIFEST.csv` | both equal | same | VERIFIED: the bars are byte-identical to the committed run's input |
| `data_manifest.py check` on the scratch folder | 2 files present and equal, plus `minute_bars_source.json` reported as different (a metadata file, not data; not investigated); 5,210 manifest files (bhavcopy, participant OI, VIX, other indices) not downloaded | | PARTIALLY VERIFIED (2 of 5,213 files) |
| Rows / trading days of the Nifty file | 886,923 / 2,370 | 886,923 / 2,368 days in dq1 coverage | VERIFIED rows; the 2-day difference in the day count is not explained (dq1 may drop two days) |
| Duplicate timestamps; timestamps strictly increasing in file order | 0; yes | | VERIFIED |
| NaN or non-positive price; OHLC inconsistent bars (high < low, open or close outside the range) | 0; 0 | | VERIFIED |
| Days with 375 bars | 2,352 of 2,370; none above 375 | median 375 | VERIFIED |
| Short days (43 to 107 bars) | 6: 2021-11-04, 2023-11-12, 2024-03-02, 2024-05-18, 2024-11-01, 2025-10-21 (Diwali muhurat and special Saturday sessions) | `data_validation` lists 5 bar days that are not official days since 2023 | consistent; they explain all 1,774 weekend bars and 121 bars outside 09:15 to 15:29; the committed run drops days under 95% coverage (`min_coverage 0.95` in run_info.json), which I did not rerun |
| Gaps inside a session | 4 gaps on 4 days | | OUR CHECK |
| Days with more than 5% / 20% zero close-to-close returns; median share | 8 / 3; 0.008 | dq1 coverage: 8 / 3; 0.008 | VERIFIED |
| Days with more than 5% / 20% bars with high = low (stricter definition) | 6 / 3 | not reported | OUR CHECK |
| Returns beyond 20 robust sd | 188 (largest 253 sd) | | OUR CHECK: heavy tails and event days, not checked bar by bar |
| Official daily comparison, 2024-05-02 to 2024-08-09 (69 NSE index days downloaded with `download_nse.py`) | Nifty 50 and Bank Nifty: open, high and low within 5 bp on 100% of days; worst high or low difference 0.0 bp; close median difference 4.9 bp (Nifty 50) and 7.6 bp (Bank Nifty), 95th percentile 19.5 and 27.7 bp | full period: highs 943 of 943, lows 942 of 943 | VERIFIED for the sample window; the full 943-day comparison was not rerun (it needs about 20 more minutes of NSE downloads) |

Status: PARTIALLY VERIFIED. What is established: the minute bars in the committed manifest are exactly what the fetch script returns today, they are internally clean, and a 69-day sample agrees with NSE's own daily values. What is not: the bhavcopy, participant OI and VIX files, and the full-period official comparison.


## A5. Sensitivity rerun of the hybrid

BLOCKED. It needs data/raw (about 25 minutes of downloads, per docs/DATA_ACQUISITION.md) and a pipeline rerun of the walk-forward; the container restarted during this work and the coordinator said to skip it when long. A synthetic substitute exists in A2 (the same code path on simulated data, 10 label shuffles). I did not redo P99 (seed bootstrap) or P106 (gain decomposition).

## Defects needing a src/ change (routed through the coordinator)

| Id | Where | Change |
| --- | --- | --- |
| A4-A-001 | `src/niftygex/volatility.py` `daily_egarch_variance`, `_egarch_ok`, `_ewma_variance` | make the model choice and the filter causal: run the sanity check and the filter on data up to the end of the current block only, floor the EWMA from the training window; add a test that replaces (not removes) the future and uses a refit block straddling the cut; rerun the pipeline and report the change in the headline |
| A4-A-002 | `src/niftygex/gex.py` `strike_gex`, `daily_summary`, `gamma_flip` | validate inputs (oi >= 0, 0 < iv < cap, T > 0, lot size > 0, boolean option type, weights in [-1, 1]) and use `sum(skipna=False)` or return NaN when the chain is invalid |

## Commands and seeds

```
cd reports/stress
python -I a1_recompute.py <repo> <repo>/reports/stress         # bootstrap seed 9,600,001
python a2_leakage.py a2_leakage.csv                              # markets 9,700,001 and 9,700,002; shuffles 9,710,000+k; permutations 9,720,000+k
python a2b_egarch_selection.py <repo> a2b_egarch_selection.csv
python a3_gex_stress.py a3_raw_cases.csv && python -I a3_classify.py   # seeds 9,800,000 and 9,800,100
```
All with OMP_NUM_THREADS=1 and the venv in the scratchpad. None of the seeds used here (9,600,001 and 9,700,001 to 9,800,100) appears in the repository code.

## What this report does not show

The raw NSE bars are not in the repository, so nothing here re-runs the committed pipeline: A1 re-derives numbers from committed forecasts, A2 and A3 use simulated data, and the effect of the A2 defect on the real headline is open. No claim is made about profit; the prototype is a forecasting study.
