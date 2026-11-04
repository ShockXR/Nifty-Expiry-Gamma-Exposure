# Extended out-of-sample backtests of the hybrid forecast (REAL DATA)

All checks in this document were run by an AI agent (Agent A of the backtest round) on the committed
forecast files. Nothing was re-estimated and no raw data was read. Every number below is REAL DATA:
it comes from the walk-forward forecasts of the real NSE run in `results/pipeline_nse_v2/` and the
paper-as-printed daily losses in `results/p116_benchmark_as_printed/`. The script is
`experiments/backtest_extended.py`; its outputs are in `results/backtest_extended/` (Bank Nifty and Fin Nifty reruns in the `banknifty/` and
`finnifty/` subfolders, see the replication section). The tests of its
helpers (SYNTHETIC arrays) are in `tests/test_backtest_extended.py`.

Models compared: the hybrid (`hybrid_base`, our mcsGARCH times a LightGBM correction), our mcsGARCH
(`mcsgarch`) and the paper-as-printed mcsGARCH (`mcs_cpp`, for which only day-level QLIKE losses are
committed, so it appears only in items 1, 4 and 6). Loss: QLIKE = y / f - log(y / f) - 1, averaged
within a day and then across the 792 test days (10 August 2023 to 2 November 2026). Gain in % is
100 (1 - mean loss of the hybrid / mean loss of the benchmark); positive means the hybrid is better.
DM = the repo's Diebold-Mariano test (Newey-West variance, Harvey-Leybourne-Newbold factor, Student
t), one-sided p that the hybrid has the lower loss unless stated otherwise.

## The answer in plain words

The headline gain (5.80% / 7.65% / 6.36% over our mcsGARCH at 15 minutes / 30 minutes / last hour,
4.43% / 6.64% / 6.16% over the paper as printed) survives every slice and every alternative loss that
is meant for noisy variance proxies. In detail:

1. **By regime (VERIFIED).** The gain over our mcsGARCH is positive in every one of the 34 cells at 15
   and 30 minutes (VIX terciles, realized-variance terciles, expiry or not, expiry weekday, weekday,
   net GEX sign under both rules, distance to the gamma flip). At 15 minutes the smallest cell gain is
   4.4% (multi-weekly expiry days and Thursday expiries under the single-weekly regime). The only
   negative cells are at the last hour on tiny or noisy subsets (5 Monday expiries; 41 Thursday
   expiries of the single-weekly-Thursday regime; 75 days close to the participant-rule flip). A
   Giacomini-White regression of the daily gain on all the conditioning variables finds no instrument
   that predicts the gain (joint Wald p = 0.34 / 0.63 / 0.43): the gain is not concentrated in a
   regime one could know in advance. Against the paper as printed the gain is smaller on expiry days
   (3.0% at 15 minutes, p = 0.05) and is not significant on the 67 multi-weekly expiry days (-0.1%).
2. **Calibration and encompassing (PARTIALLY VERIFIED).** On the log scale the hybrid is well
   calibrated at 15 and 30 minutes (Mincer-Zarnowitz slope 0.997, standard error 0.011 to 0.014),
   while mcsGARCH's slope is 0.94 (p for slope = 1 below 1e-4). Both under-forecast the level a little:
   the mean of y / f is 1.05 (hybrid) and 1.08 (mcsGARCH) at 15 minutes. On the variance scale both
   are over-dispersed (slope well below 1) and that result is driven by a handful of extreme days. The
   hybrid encompasses mcsGARCH: the optimal QLIKE weight on the hybrid is 1.11 / 1.20 / 1.47, and
   moving from weight 1 to the optimum gains only 0.08% / 0.31% / 0.88% (DM p 0.22 / 0.06 / 0.13).
   mcsGARCH adds nothing positive to the hybrid.
3. **Alternative losses (VERIFIED for proxy-robust losses, FAILED for MSE).** The hybrid beats mcsGARCH
   with p below 0.004 under QLIKE, MSE on log variance, QLIKE on the volatility scale and Patton's
   b = -1 loss. Under MSE on the variance scale (Patton b = 0) and b = 1 the point gains are large
   (18% to 48%) but not significant (p 0.05 to 0.15), because those losses are dominated by a few
   days in June 2024. This is the expected behaviour of MSE on a heavy-tailed proxy, not evidence
   against the hybrid, but a reader who insists on MSE should treat the gain as unproven.
4. **Data snooping across the ladder (VERIFIED).** Hansen's SPA test with the stationary bootstrap
   (5,000 draws, mean block 10 days, seed 9700001) rejects "no model in the 32-model ladder beats our
   mcsGARCH" with p = 0.000 / 0.000 / 0.005. 26 of the 32 alternatives (81%) have a lower mean loss
   than mcsGARCH. With the hybrid as the benchmark, no model beats it: SPA p = 0.92 / 0.16 / 0.82.
5. **Horizon consistency (VERIFIED).** With the same information set, the hybrid's direct 30-minute
   forecast and twice its 15-minute forecast at the same anchor have the same QLIKE (0.2009 against
   0.2009, p = 0.99) and a median ratio of 0.998: the hybrid is internally consistent. The direct
   last-hour forecast beats 2 x the 30-minute forecast made at 14:30 by 6.0% (p = 0.005) and 4 x the
   15-minute forecast by 18.6% (p = 0.001), because variance rises into the close and the last-hour
   model knows that. Summing two 15-minute forecasts beats the direct 30-minute forecast by 47%, but
   that sum uses a forecast made 15 minutes later, so it is not a fair race and is reported only as a
   consistency check (median ratio direct / sum = 1.03 for the hybrid, 1.05 for mcsGARCH).
6. **Rolling view (VERIFIED).** Every one of the 730 rolling 63-day windows shows a positive gain
   over mcsGARCH at 15 and 30 minutes (minimum 1.55% and 1.57%); at the last hour 85% of windows are
   positive (minimum -8.3%). The longest run of consecutive days on which the hybrid loses to
   mcsGARCH is 6 / 5 / 8 days. The largest drawdown of the cumulative gain is 1.4% / 1.5% / 8.5% of
   the total gain accumulated over the test period.

Overall status: VERIFIED for items 1, 4, 5, 6; PARTIALLY VERIFIED for item 2 (log-scale
calibration good, variance-scale calibration poor for both models and outlier-driven); item 3
VERIFIED under the proxy-robust losses and FAILED under MSE-type losses. Nothing was found that
undermines the headline, and one feature of the forecasts (over-dispersion on the variance scale,
which the hybrid reduces but does not remove) is worth a sentence in the paper.

## 1. Conditional gains by regime

File: `results/backtest_extended/regime_gains.csv` (every cell, both benchmarks, with n, mean losses,
gain, DM statistic, one-sided DM p and the share of days the hybrid wins). Conditioning variables
are all known at the close of the previous day: previous-day India VIX and previous-day realized
variance (terciles over the 792 test days, from `daily_market.csv`), the expiry flag and the weekday
of the date, the sign of the previous day's net GEX from `daily_gex_panel.csv` (row published at the
close of day t-1, which is what the pipeline's features use), and the distance between the previous
close and the previous day's gamma flip. The weekly expiry regimes come from `daily_market.csv`:
multi_weekly to 19 November 2024, single_weekly_thu to 29 August 2025, single_weekly_tue from
1 September 2025.

Gain of the hybrid in % (one-sided DM p in brackets), hybrid versus our mcsGARCH:

| Family | Cell | n days | next15 | next30 | last hour |
| --- | --- | --- | --- | --- | --- |
| all | all | 792 | 5.80 (1.8e-9) | 7.65 (3.6e-15) | 6.36 (1.0e-4) |
| previous-day VIX tercile | low | 264 | 6.10 (0.006) | 8.02 (3.5e-5) | 6.08 (1.5e-4) |
|  | mid | 264 | 6.40 (1.2e-7) | 7.08 (3.3e-7) | 5.70 (0.063) |
|  | high | 264 | 4.86 (3.8e-10) | 7.88 (4.8e-7) | 7.29 (0.044) |
| previous-day RV tercile | low | 264 | 5.05 (6.9e-6) | 6.61 (1.6e-6) | 5.31 (0.002) |
|  | mid | 264 | 5.58 (8.6e-8) | 7.52 (1.6e-8) | 9.19 (0.024) |
|  | high | 264 | 6.75 (6.0e-5) | 8.83 (6.3e-7) | 5.69 (0.016) |
| expiry | expiry day | 169 | 6.64 (0.001) | 6.23 (2.1e-4) | 7.91 (0.010) |
|  | non-expiry day | 623 | 5.50 (2.2e-14) | 8.13 (1.5e-15) | 5.95 (4.6e-4) |
| expiry weekday | Tuesday expiry | 56 | 10.82 (0.036) | 10.09 (0.022) | 14.34 (0.012) |
|  | Thursday expiry | 104 | 4.94 (1.7e-6) | 5.45 (1.7e-4) | 8.81 (0.008) |
|  | Monday expiry (holiday shifts) | 5 | 6.45 (0.009) | 2.41 (0.30) | -16.71 (0.81) |
| regime x expiry | multi_weekly, expiry | 67 | 4.44 (0.002) | 5.40 (0.021) | 12.79 (0.030) |
|  | multi_weekly, non-expiry | 245 | 5.46 (4.0e-12) | 7.18 (2.6e-5) | 8.79 (0.049) |
|  | single_weekly_thu, expiry | 41 | 4.38 (0.001) | 4.15 (0.003) | -2.38 (0.73) |
|  | single_weekly_thu, non-expiry | 151 | 5.45 (3.2e-6) | 7.27 (1.0e-10) | 1.20 (0.29) |
|  | single_weekly_tue, expiry | 61 | 10.58 (0.031) | 9.49 (0.021) | 13.14 (0.019) |
|  | single_weekly_tue, non-expiry | 227 | 5.59 (2.4e-4) | 9.80 (5.1e-7) | 6.21 (0.001) |
| weekday | Mon | 158 | 4.98 (0.002) | 6.99 (7.3e-4) | 2.04 (0.22) |
|  | Tue | 159 | 6.17 (0.009) | 8.21 (5.9e-4) | 8.89 (0.015) |
|  | Wed | 158 | 6.31 (9.4e-7) | 8.01 (3.3e-8) | 5.54 (0.073) |
|  | Thu | 160 | 4.78 (6.8e-8) | 6.52 (1.1e-6) | 8.93 (0.002) |
|  | Fri | 157 | 6.95 (4.1e-12) | 8.68 (2.8e-9) | 11.61 (0.013) |
| net GEX, US convention (prev. day) | positive | 697 | 5.27 (2.4e-19) | 7.33 (2.9e-17) | 8.45 (6.4e-5) |
|  | negative | 95 | 8.99 (0.048) | 9.88 (0.012) | 1.41 (0.27) |
| net GEX, participant rule (prev. day) | positive | 461 | 5.62 (5.7e-5) | 7.69 (2.8e-10) | 5.03 (0.002) |
|  | negative | 331 | 6.07 (3.7e-11) | 7.59 (2.6e-7) | 9.42 (0.009) |
| prev. close vs US-convention flip | within 0.5% | 730 | 5.50 (2.1e-16) | 7.26 (4.0e-19) | 6.36 (1.5e-4) |
|  | farther | 62 | 8.47 (0.045) | 11.20 (0.008) | 6.43 (0.091) |
| prev. close vs participant flip (355 days have one) | within 0.5% | 75 | 6.16 (3.9e-9) | 8.37 (3.4e-10) | -2.64 (0.71) |
|  | farther | 280 | 6.68 (4.4e-8) | 7.81 (1.1e-7) | 7.98 (0.028) |
| distance to US-convention flip, tercile | near | 264 | 5.32 (2.0e-6) | 7.44 (1.1e-7) | 5.60 (0.007) |
|  | mid | 264 | 6.84 (6.6e-9) | 8.14 (3.4e-9) | 10.72 (0.001) |
|  | far | 264 | 5.35 (4.6e-4) | 7.41 (3.3e-6) | 2.78 (0.21) |

Reading: at 15 and 30 minutes every cell is positive and almost every cell is significant on its
own. The Tuesday-expiry cells look larger (about 10%) than the Thursday-expiry cells (about 5%), but
the difference is within noise (the GW slope test on the expiry and weekday dummies below does not
reject). The US-convention flip sits within 0.5% of the close on 730 of 792 days, so that cut is
nearly empty on one side; the tercile cut is the informative one and shows no pattern.

Hybrid versus the paper as printed (`mcs_cpp`), the cells where the picture differs:

| Cell | n days | next15 | next30 | last hour |
| --- | --- | --- | --- | --- |
| all | 792 | 4.43 (1.3e-10) | 6.64 (1.6e-16) | 6.16 (7.8e-5) |
| expiry day | 169 | 2.97 (0.050) | 4.35 (0.006) | 7.88 (0.024) |
| non-expiry day | 623 | 4.92 (3.3e-19) | 7.42 (5.8e-18) | 5.70 (3.2e-4) |
| multi_weekly, expiry | 67 | -0.07 (0.52) | 3.18 (0.12) | 6.31 (0.21) |
| single_weekly_thu, expiry | 41 | 2.22 (0.14) | 3.05 (0.095) | 0.03 (0.50) |
| single_weekly_tue, expiry | 61 | 6.56 (0.073) | 7.27 (0.027) | 16.73 (0.022) |
| Thursday expiry | 104 | 1.28 (0.15) | 3.65 (0.019) | 5.15 (0.11) |
| weekday Thu | 160 | 1.87 (0.033) | 4.59 (8.1e-4) | 4.78 (0.077) |

Reading: the paper-as-printed model has its own expiry-day treatment, and on the 108 expiry days
before September 2025 (Thursday expiries) the hybrid's edge over it at 15 minutes is small and not
significant. The headline over `mcs_cpp` is carried by non-expiry days and by the Tuesday-expiry
regime.

Giacomini-White conditional predictive ability (file `results/backtest_extended/gw_tests.csv`,
coefficients in `gw_slope_coefficients.csv`). The code is the `gw_test` and `slope_wald` functions of
`experiments/referee_forecast_gw.py` (which already used a constant, the lagged daily gain, the expiry
dummy and regime dummies); only the instruments are new. Each row adds one family of instruments to
the constant and the lagged gain. The GW chi-square tests whether the expected gain is zero
conditional on the instruments; it rejects everywhere because the unconditional mean gain is far
from zero (this is the known result). The useful column is the slope Wald: do the new instruments
predict the day's gain? Hybrid versus our mcsGARCH, 791 days:

| Instrument family (added to constant + lagged gain) | next15 slope chi2 (df), p | next30 | last hour |
| --- | --- | --- | --- |
| none: the lagged gain alone | 3.97 (1), 0.046 | 0.78 (1), 0.38 | 0.01 (1), 0.91 |
| VIX tercile dummies | 1.83 (2), 0.40 | 0.05 (2), 0.98 | 2.91 (2), 0.23 |
| RV tercile dummies | 0.01 (2), 0.99 | 0.43 (2), 0.81 | 0.60 (2), 0.74 |
| expiry dummy | 1.32 (1), 0.25 | 0.02 (1), 0.88 | 0.20 (1), 0.66 |
| weekday dummies | 1.80 (4), 0.77 | 0.81 (4), 0.94 | 1.55 (4), 0.82 |
| net GEX sign, US convention | 0.77 (1), 0.38 | 0.46 (1), 0.50 | 0.32 (1), 0.57 |
| net GEX sign, participant rule | 0.00 (1), 0.98 | 0.00 (1), 0.95 | 0.04 (1), 0.84 |
| within 0.5% of US flip | 0.45 (1), 0.50 | 1.11 (1), 0.29 | 0.06 (1), 0.81 |
| distance-to-flip tercile dummies | 2.69 (2), 0.26 | 0.12 (2), 0.94 | 5.02 (2), 0.081 |
| all of the above | 15.63 (14), 0.34 | 11.67 (14), 0.63 | 14.29 (14), 0.43 |
| GW chi2 of "all of the above" (16 df) | 112.0, p < 1e-15 | 120.0, p < 1e-15 | 27.7, p = 0.034 |

Against `mcs_cpp` the joint slope p is 0.23 / 0.90 / 0.72 (same file). No instrument family predicts
the gain at the 5% level in 27 tests, except the lagged gain at 15 minutes (p = 0.046, coefficient
0.30 points), which is what one expects from mild persistence in daily losses.

## 2. Calibration and encompassing

Files: `results/backtest_extended/calibration_mincer_zarnowitz.csv`, `encompassing.csv`,
`ratio_y_over_f.csv`. Regressions are pooled over anchors (19,800 / 19,008 / 792 observations) with a
HAC covariance that first sums the scores within each day and then applies Newey-West across days
with the repo's bandwidth rule. "variance, trimmed" drops the 8 days whose mean realized variance is
in the top 1% (the June 2024 election days are among them).

Mincer-Zarnowitz, y on f (slope, standard error, p for slope = 1; the log scale regresses log y on
log f):

| Horizon | Model | log: slope (se), p slope=1 | log intercept | variance: slope (se) | variance, trimmed: slope (se), p slope=1 |
| --- | --- | --- | --- | --- | --- |
| next15 | hybrid | 0.997 (0.011), 0.80 | -0.26 | 0.11 (0.03) | 0.63 (0.17), 0.026 |
| next15 | mcsGARCH | 0.938 (0.013), 2.9e-6 | -1.07 | 0.08 (0.02) | 0.26 (0.14), 2.5e-7 |
| next15 | lgbm_base | 1.040 (0.013), 0.001 | 0.33 | 1.40 (0.26) | 1.08 (0.04), 0.037 |
| next15 | HAR | 1.003 (0.012), 0.81 | -0.22 | 1.11 (0.17) | 0.92 (0.05), 0.073 |
| next30 | hybrid | 0.997 (0.014), 0.84 | -0.21 | 0.12 (0.03) | 0.57 (0.17), 0.012 |
| next30 | mcsGARCH | 0.940 (0.015), 5.6e-5 | -0.95 | 0.09 (0.03) | 0.30 (0.15), 4.6e-6 |
| next30 | lgbm_base | 1.054 (0.015), 3.1e-4 | 0.54 | 1.31 (0.26) | 1.02 (0.03), 0.43 |
| last hour | hybrid | 0.875 (0.043), 0.003 | -1.65 | 0.76 (0.08) | 0.55 (0.10), 5.2e-6 |
| last hour | mcsGARCH | 0.796 (0.041), 5.0e-7 | -2.61 | 0.68 (0.12) | 0.43 (0.09), 6.5e-10 |
| last hour | lgbm_base | 0.973 (0.048), 0.57 | -0.45 | 1.07 (0.19) | 0.82 (0.06), 0.004 |

The joint test of intercept 0 and slope 1 on the log scale rejects for every model (p below 1e-40
at 15 and 30 minutes) because the intercept is negative by construction: the log of a noisy variance
proxy sits below the log of its mean, so a perfectly calibrated variance forecast has a negative
log-scale intercept. The slope is the informative number. On the variance scale both the hybrid and
mcsGARCH have slopes far below 1: when they forecast a very high variance, the realized variance is
on average much lower (over-dispersed forecasts). The learner alone (lgbm_base) and HAR do not have
this problem. The untrimmed variance-scale slopes of 0.08 to 0.12 are set by a few days; the trimmed
ones (0.26 for mcsGARCH, 0.63 for the hybrid) show the hybrid removes about half of the problem.

Distribution of y / f over all forecast windows:

| Horizon | Model | mean | p10 | median | p90 | share y/f > 1 | mean log(y/f) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| next15 | hybrid | 1.051 | 0.36 | 0.79 | 1.80 | 0.35 | -0.217 |
| next15 | mcsGARCH | 1.078 | 0.36 | 0.80 | 1.86 | 0.36 | -0.207 |
| next15 | lgbm_base | 1.052 | 0.36 | 0.78 | 1.80 | 0.34 | -0.223 |
| next30 | hybrid | 1.032 | 0.43 | 0.82 | 1.72 | 0.35 | -0.169 |
| next30 | mcsGARCH | 1.055 | 0.42 | 0.82 | 1.78 | 0.36 | -0.162 |
| last hour | hybrid | 1.117 | 0.47 | 0.86 | 1.84 | 0.40 | -0.107 |
| last hour | mcsGARCH | 1.142 | 0.45 | 0.88 | 1.90 | 0.41 | -0.097 |

A calibrated variance forecast has mean y / f = 1; the median is below 1 for any right-skewed proxy.
Both models under-forecast the level slightly (the hybrid by 5% / 3% / 12%, mcsGARCH by 8% / 6% / 14%).
The 10th to 90th percentile band of 0.36 to 1.80 at 15 minutes is the proxy noise of a 15-minute
realized variance, not forecast error alone.

Encompassing, y = w f_hybrid + (1 - w) f_mcsGARCH + intercept (w = 1 means mcsGARCH adds nothing):

| Horizon | Scale | w (se) | p for w = 0 | p for w = 1 | QLIKE-optimal w (grid) | gain of optimum over hybrid, % (DM p) | gain over mcsGARCH, % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| next15 | log | 1.14 (0.06) | 2e-88 | 0.015 | 1.11 | 0.08 (0.22) | 5.87 |
| next15 | variance, trimmed | 1.54 (0.04) | 0 | 6e-47 |  |  |  |
| next30 | log | 1.17 (0.07) | 7e-60 | 0.016 | 1.20 | 0.31 (0.064) | 7.94 |
| next30 | variance, trimmed | 1.90 (0.08) | 2e-122 | 9e-29 |  |  |  |
| last hour | log | 1.33 (0.19) | 3e-12 | 0.085 | 1.47 | 0.88 (0.13) | 7.18 |
| last hour | variance, trimmed | 2.30 (0.31) | 7e-14 | 2e-5 |  |  |  |

Reading: the weight on the hybrid is at or above 1 on every scale, so mcsGARCH never receives a
positive weight; the regressions say one should extrapolate past the hybrid away from mcsGARCH, but
the QLIKE improvement from doing so is tiny and not significant. The hybrid encompasses mcsGARCH;
mcsGARCH does not add anything to the hybrid. (The untrimmed variance-scale weights of 3.9 / 5.0 / 2.2
are in the file and are set by the same handful of extreme days.)

## 3. Alternative loss functions

File: `results/backtest_extended/alternative_losses.csv`. Gain of the hybrid over our mcsGARCH, %,
with the one-sided DM p. "Share" is the share of the 792 days on which the hybrid has the lower daily
loss. Patton's family: b = -2 is QLIKE, b = 0 is MSE / 2 (same ranking and same DM as MSE), b = -1
and b = 1 are the other two members asked for.

| Loss | next15: gain (p), share | next30 | last hour |
| --- | --- | --- | --- |
| QLIKE (Patton b = -2) | 5.80 (1.8e-9), 0.71 | 7.65 (3.6e-15), 0.74 | 6.36 (1.0e-4), 0.54 |
| MSE on log variance | 5.21 (2.5e-12), 0.69 | 6.58 (2.4e-13), 0.70 | 5.86 (2.1e-4), 0.54 |
| QLIKE on the volatility scale | 5.07 (9.6e-19), 0.71 | 6.56 (6.6e-21), 0.73 | 5.97 (5.6e-5), 0.54 |
| Patton b = -1 | 10.75 (0.004), 0.76 | 12.03 (0.001), 0.75 | 8.42 (1.4e-4), 0.54 |
| MSE on variance (Patton b = 0) | 33.21 (0.13), 0.72 | 26.91 (0.13), 0.75 | 18.27 (0.053), 0.54 |
| Patton b = 1 | 47.56 (0.15), 0.70 | 38.06 (0.15), 0.74 | 34.27 (0.12), 0.54 |

Reading: under the four losses that are robust to proxy noise and not dominated by outliers the
gain is 5% to 12% with p at most 0.004. Under MSE on the variance scale and b = 1 the hybrid wins on
the same share of days, the point gain is large, but the DM p is 0.05 to 0.15: these losses are
dominated by the June 2024 days, where both mcsGARCH-based forecasts were far too high. The same
file shows that under MSE the learner alone (`lgbm_base`) has a far lower loss than the hybrid, which
is the over-dispersion of item 2 seen through a loss that punishes it.

## 4. Data snooping across the ladder: Hansen's SPA test

Files: `results/backtest_extended/spa.csv`, `spa_per_model.csv`. Daily QLIKE of the 32 ladder
models in the forecast files plus `mcs_cpp` from the P116 daily losses (33 models, 792 days).
Stationary bootstrap with mean block 10 days, 5,000 draws, seed 9700001 (the same block length as
the bootstrap intervals in `reports/BACKTEST_STRESS_REPORT.md`). The SPA statistic is the largest
studentized mean loss difference; the three p-values are Hansen's consistent, lower and upper
versions. White's reality-check p from the same draws is reported for completeness; it is not
studentized and is dominated by the very noisy naive models, so it is uninformative here (0.68 /
0.54 / 0.76).

| Benchmark | Horizon | alternatives with lower mean loss | best alternative by t (gain, %) | best by gain (gain, %) | SPA statistic | SPA p consistent (lower, upper) |
| --- | --- | --- | --- | --- | --- | --- |
| our mcsGARCH | next15 | 26 of 32 (81%) | hybrid_v1gex (5.56) | hybrid_gex_us (5.97) | 8.13 | 0.000 (0.000, 0.000) |
| our mcsGARCH | next30 | 26 of 32 (81%) | hybrid_base (7.65) | hybrid_v1gex (8.10) | 7.84 | 0.000 (0.000, 0.000) |
| our mcsGARCH | last hour | 26 of 32 (81%) | hybrid_gex_placebo (5.81) | linear_v1 (8.46) | 3.67 | 0.005 (0.003, 0.005) |
| hybrid_base | next15 | 2 of 32 (6%) | hybrid_gex_us (0.18) | hybrid_gex_us (0.18) | 0.63 | 0.923 (0.646, 0.961) |
| hybrid_base | next30 | 6 of 32 (19%) | hybrid_gex_gross (0.42) | hybrid_v1gex (0.49) | 1.87 | 0.159 (0.130, 0.266) |
| hybrid_base | last hour | 8 of 32 (25%) | linear_gex_client (1.99) | linear_v1 (2.24) | 0.89 | 0.817 (0.549, 0.905) |

Reading: the claim "something in the ladder beats mcsGARCH" is not a product of searching over 32
models; the SPA p is 0.000 at 15 and 30 minutes with 5,000 draws, and 0.005 at the last hour. In the
other direction nothing in the ladder beats the hybrid: the best alternative at 15 minutes is a GEX
variant that is 0.18% better, well inside noise. The six models that do not beat mcsGARCH are the
two naive rules, VIX, option-implied, HAR and the 63-day-refit mcsGARCH; `mcs_cpp` beats our
mcsGARCH by 1.43% / 1.08% / 0.22%.

## 5. Horizon consistency

File: `results/backtest_extended/horizon_consistency.csv`. Realized variance is additive, so the
30-minute window at anchor a equals the two 15-minute windows at a and a + 15 (checked to 1e-9), and
the last hour (anchor 315) equals the 30-minute windows at 315 and 345 or the four 15-minute windows
from 315. Two kinds of comparison, both with the DM test (two-sided p):

* same information: the direct forecast against 2 x the 15-minute forecast at the same anchor
  (or 2 x the 30-minute and 4 x the 15-minute forecast at 14:30); this is a fair race;
* sum of the shorter forecasts: the later ones are made after the window has started, so they
  know more; this measures internal consistency only.

| Comparison | Model | QLIKE direct | QLIKE alternative | direct better by, % (p) | median ratio direct / alternative (p10, p90) |
| --- | --- | --- | --- | --- | --- |
| next30 vs 2 x next15 at the same anchor | hybrid | 0.2009 | 0.2009 | 0.01 (0.99) | 0.998 (0.84, 1.14) |
| next30 vs 2 x next15 at the same anchor | mcsGARCH | 0.2176 | 0.2158 | -0.81 (0.47) | 1.000 (0.88, 1.12) |
| next30 vs sum of the two next15 (second made later) | hybrid | 0.2009 | 0.1364 | -47.3 (<1e-30) | 1.034 (0.86, 1.16) |
| next30 vs sum of the two next15 (second made later) | mcsGARCH | 0.2176 | 0.1377 | -58.0 (<1e-30) | 1.047 (0.84, 1.17) |
| last hour vs 2 x next30 at 14:30 | hybrid | 0.2243 | 0.2386 | 5.98 (0.005) | 1.029 (0.94, 1.16) |
| last hour vs 2 x next30 at 14:30 | mcsGARCH | 0.2395 | 0.2533 | 5.46 (0.002) | 1.016 (0.96, 1.11) |
| last hour vs 4 x next15 at 14:30 | hybrid | 0.2243 | 0.2755 | 18.6 (0.001) | 1.163 (1.01, 1.36) |
| last hour vs 4 x next15 at 14:30 | mcsGARCH | 0.2395 | 0.2962 | 19.2 (0.001) | 1.149 (1.05, 1.32) |
| last hour vs sum of two next30 (second made later) | hybrid | 0.2243 | 0.1882 | -19.2 (0.003) | 1.038 (0.84, 1.19) |
| last hour vs sum of four next15 (later ones made later) | hybrid | 0.2243 | 0.1611 | -39.2 (2e-6) | 1.058 (0.77, 1.31) |

Reading: the hybrid's 30-minute forecast is the same as twice its 15-minute forecast at the same
anchor, both in level (median ratio 0.998) and in accuracy (p = 0.99), so the two horizons tell one
story. For the last hour the direct model forecasts about 16% more variance than four times the
15-minute forecast made at 14:30, and it is right to: the direct forecast is 19% better. That
matches the rising variance into the close. The hybrid's direct forecasts also sit closer to the sums
than mcsGARCH's do (ratio 1.03 against 1.05 at 30 minutes), so the hybrid is at least as consistent
across horizons as the model it corrects. "Which is better" between a direct 30-minute forecast and
a sum of 15-minute forecasts cannot be answered fairly from these files, because the second
15-minute forecast is made with 15 more minutes of data; the 47% gap is the value of that data.

## 6. Rolling 63-day gain and runs of losing days

Files: `results/backtest_extended/rolling_gain_summary.csv`, `rolling_gain_series.csv` (730
windows). Gain within each window of 63 consecutive test days; "losing day" means the hybrid's daily
QLIKE is above the benchmark's. The drawdown is measured on the cumulative sum of daily gains in
points of the benchmark's mean daily QLIKE (100 points = one day's mean loss).

| Benchmark | Horizon | rolling gain min, % (window end) | median, % | max, % | windows positive | longest losing run (dates) | longest winning run | max drawdown, points (trough) | cumulative gain, points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| our mcsGARCH | next15 | 1.55 (2025-07-28) | 5.87 | 11.03 | 730 of 730 | 6 (2026-10-15 to 10-23) | 14 | 66 (2025-09-09) | 4,593 |
| our mcsGARCH | next30 | 1.57 (2024-09-10) | 7.27 | 13.53 | 730 of 730 | 5 (2023-09-26 to 10-03) | 17 | 91 (2024-06-18) | 6,058 |
| our mcsGARCH | last hour | -8.30 (2024-09-02) | 6.51 | 20.21 | 621 of 730 (85%) | 8 (2023-09-06 to 09-15) | 9 | 428 (2025-05-15) | 5,039 |
| paper as printed | next15 | 0.52 (2024-06-18) | 4.87 | 7.96 | 730 of 730 | 5 (2025-02-12 to 02-18) | 13 | 88 (2026-08-17) | 3,509 |
| paper as printed | next30 | 0.56 (2024-09-12) | 6.45 | 11.90 | 730 of 730 | 4 (2023-12-04 to 12-07) | 28 | 92 (2024-06-18) | 5,263 |
| paper as printed | last hour | -4.84 (2024-09-02) | 6.33 | 18.35 | 630 of 730 (86%) | 6 (2023-12-05 to 12-12) | 15 | 296 (2024-07-25) | 4,877 |

Reading: at 15 and 30 minutes there is no quarter in which the hybrid lost to either benchmark; the
worst 63-day window still gained 1.5% over our mcsGARCH and 0.5% over the paper as printed. The
hybrid never lost more than six days in a row at 15 minutes. The last hour is the weak horizon, as
every earlier report says: one forecast per day, 54% of days won, and a 63-day window in mid-2024
with an 8% loss.

## Does it replicate on Bank Nifty and Fin Nifty?

The same script was run unchanged (`--index banknifty`, `--index finnifty`) on the committed runs in
`results/pipeline_nse_banknifty/` (660 test days, 10 August 2023 to 22 April 2026) and
`results/pipeline_nse_finnifty/` (604 test days, to 27 January 2026). Outputs are in
`results/backtest_extended/banknifty/` and `results/backtest_extended/finnifty/` (same 13 files). All
REAL DATA, run by an AI agent. Not supported and therefore skipped: the paper-as-printed model at
the last hour (only `daily_losses_{banknifty,finnifty}_{next15,next30}.csv` are committed), so
`mcs_cpp` appears only at 15 and 30 minutes for these indices and the last-hour SPA has 31
alternatives. The regime labels in these runs are the Nifty calendar periods; the expiry-weekday
cells use the actual date, which is why Bank Nifty has a Wednesday-expiry cell. Same seed (9700001),
draws and block length as above. Unless stated, numbers are hybrid versus our mcsGARCH; "p" is the
one-sided DM p, and "< 1e-4" means the file shows 0.0000.

Short answer: at 15 and 30 minutes everything replicates on both indices (positive in every regime
cell, log-scale calibration slope at 1, significant under every proxy-robust loss, SPA p = 0.000,
every rolling 63-day window positive or within 0.1% of it). The last hour replicates on Bank Nifty
(8.1%, p = 3e-7) but not on Fin Nifty (1.0%, p = 0.30), where the hybrid also loses on expiry
cells and mcsGARCH keeps a positive encompassing weight. One new finding: on Bank Nifty at 15
minutes the SPA test with the hybrid as benchmark gives p = 0.027, so some GEX or implied-volatility
variant in the ladder beats the Bank Nifty hybrid by more than noise (best by gain
`hybridiv_gex_us`, 2.7%), which was not the case on Nifty.

Bank Nifty (660 days; files in `results/backtest_extended/banknifty/`):

| Check | next15 | next30 | last hour |
| --- | --- | --- | --- |
| Gain over our mcsGARCH, % (p) | 3.92 (2.6e-12) | 5.03 (2.0e-11) | 8.11 (3.2e-7) |
| Gain over paper as printed, % (p) | 4.73 (7.3e-13) | 6.20 (5.0e-13) | not committed |
| Regime cells positive (significant at 5%) of 33 | 33 (29) | 33 (28) | 28 (24) |
| Smallest cell, % (cell, n, p) | 1.78 (Thursday expiries, 18, 0.17) | 0.34 (single-weekly-Tue expiries, 7, 0.46) | -10.18 (Wednesday expiries, 54, 0.88) |
| Other negative cells | none | none | expiry days -3.0 (84), multi-weekly expiries -9.1 (67), Thursday expiries -1.7 (18), near participant flip -0.7 (47) |
| GW joint slope p (all instruments, 14 df) | 0.56 | 0.61 | 0.021 (expiry-related instruments predict the gain) |
| Log-scale MZ slope hybrid (se) / mcsGARCH (se) | 1.018 (0.012) / 0.987 (0.014) | 1.005 (0.016) / 0.991 (0.019) | 0.949 (0.042) / 0.944 (0.048) |
| Mean y / f hybrid / mcsGARCH | 0.98 / 0.98 | 0.97 / 0.96 | 0.93 / 0.87 |
| Trimmed variance-scale slope hybrid / mcsGARCH | 0.92 / 0.74 | 0.83 / 0.71 | 0.64 / 0.57 |
| Encompassing: log w (se), p for w = 1; QLIKE-optimal w, extra gain % (p) | 1.09 (0.06), 0.11; 1.00, 0.00 (0.50) | 1.09 (0.07), 0.19; 1.08, 0.04 (0.33) | 0.98 (0.18), 0.89; 1.22, 0.40 (0.20) |
| Gain under log MSE / vol-scale QLIKE / Patton b = -1, % (p) | 5.04 (< 1e-4) / 4.47 (< 1e-4) / 3.98 (0.002) | 7.45 (< 1e-4) / 6.28 (< 1e-4) / 5.02 (< 1e-4) | 13.02 (< 1e-4) / 10.67 (< 1e-4) / 7.45 (0.002) |
| Gain under MSE on variance / Patton b = 1, % (p) | 9.83 (0.12) / 21.68 (0.15) | -1.01 (0.63) / -7.68 (0.83) | 8.78 (0.074) / 15.92 (0.14) |
| SPA, benchmark mcsGARCH: alternatives better, p consistent (lower, upper) | 24 of 32, 0.000 (0.000, 0.000) | 25 of 32, 0.000 (0.000, 0.000) | 26 of 31, 0.000 (0.000, 0.000) |
| SPA, benchmark hybrid: alternatives better, best by gain (%), p consistent (lower, upper) | 12 of 32, hybridiv_gex_us (2.72), 0.027 (0.025, 0.035) | 10 of 32, hybridiv_gex_participant (3.18), 0.205 (0.149, 0.279) | 15 of 31, linear_gex_us (4.37), 0.519 (0.401, 0.656) |
| Horizon: direct vs 2 x shorter at the same anchor, % (p), median ratio | -1.78 (0.032), 1.000 | last hour vs 2 x next30: 0.03 (0.99), 1.04 | last hour vs 4 x next15: 2.43 (0.54), 1.19 |
| Rolling 63-day gain: min % (window end), windows positive, longest losing run | 0.64 (2026-04-17), 598 of 598, 7 days | 0.41 (2026-04-17), 598 of 598, 7 days | -4.88 (2024-03-04), 545 of 598 (91%), 6 days |
| Rolling vs paper as printed: min %, windows positive | 0.86, 598 of 598 | 0.79, 598 of 598 | not committed |

Fin Nifty (604 days; files in `results/backtest_extended/finnifty/`):

| Check | next15 | next30 | last hour |
| --- | --- | --- | --- |
| Gain over our mcsGARCH, % (p) | 4.90 (3.3e-12) | 6.09 (2.2e-12) | 0.96 (0.30) |
| Gain over paper as printed, % (p) | 4.03 (4.1e-8) | 5.18 (8.2e-10) | not committed |
| Regime cells positive (significant at 5%) of 32 | 32 (27) | 32 (28) | 18 (3) |
| Smallest cell, % (cell, n, p) | 2.48 (multi-weekly expiries, 67, 0.18) | 3.04 (multi-weekly expiries, 67, 0.17) | -88.1 (Thursday expiries, 8, 0.99) |
| Other negative cells | none | none | 13 more, among them expiry days -4.4 (82), Tuesday expiries -5.3 (70), weekday Thu -8.9 (122), VIX mid tercile -3.8 (201) |
| Versus paper as printed: cells positive of 32 | 28 (negatives are the four expiry cells, -0.8 to -3.3, p 0.62 to 0.88) | 28 (same four cells, -0.7 to -1.7) | not committed |
| GW joint slope p (all instruments, 14 df) | 0.58 | 0.59 | 0.11 |
| Log-scale MZ slope hybrid (se) / mcsGARCH (se) | 0.994 (0.016) / 0.924 (0.015) | 0.983 (0.021) / 0.923 (0.018) | 0.856 (0.042) / 0.829 (0.042) |
| Mean y / f hybrid / mcsGARCH | 0.99 / 1.02 | 0.98 / 0.99 | 0.98 / 0.93 |
| Trimmed variance-scale slope hybrid / mcsGARCH | 0.81 / 0.57 | 0.78 / 0.60 | 0.47 / 0.39 |
| Encompassing: log w (se), p for w = 1; QLIKE-optimal w, extra gain % (p) | 1.07 (0.05), 0.13; 0.95, 0.02 (0.33) | 1.07 (0.06), 0.29; 1.01, 0.00 (0.48) | 0.54 (0.20), 0.024; 0.63, 0.69 (0.18) |
| Gain under log MSE / vol-scale QLIKE / Patton b = -1, % (p) | 4.88 (< 1e-4) / 4.71 (< 1e-4) / 17.48 (0.12) | 7.08 (< 1e-4) / 6.34 (< 1e-4) / 11.87 (0.084) | 5.86 (4e-4) / 3.48 (0.021) / 5.89 (0.046) |
| Gain under MSE on variance / Patton b = 1, % (p) | 43.51 (0.16) / 57.54 (0.16) | 26.89 (0.16) / 37.48 (0.16) | 16.79 (0.071) / 23.04 (0.12) |
| SPA, benchmark mcsGARCH: alternatives better, p consistent (lower, upper) | 27 of 32, 0.000 (0.000, 0.000) | 26 of 32, 0.000 (0.000, 0.000) | 20 of 31, 0.471 (0.422, 0.622) |
| SPA, benchmark hybrid: alternatives better, best by gain (%), p consistent (lower, upper) | 6 of 32, hybridiv_gex_gross (0.26), 0.942 (0.625, 0.987) | 5 of 32, hybridiv_base (0.27), 0.935 (0.730, 0.980) | 16 of 31, hybridiv_base (3.60), 0.590 (0.474, 0.701) |
| Horizon: direct vs 2 x shorter at the same anchor, % (p), median ratio | -0.61 (0.45), 1.001 | last hour vs 2 x next30: -0.35 (0.85), 1.05 | last hour vs 4 x next15: 4.68 (0.20), 1.16 |
| Rolling 63-day gain: min % (window end), windows positive, longest losing run | 0.12 (2024-06-24), 542 of 542, 5 days | -0.12 (2024-06-25), 540 of 542 (99.6%), 4 days | -18.44 (2024-10-15), 318 of 542 (59%), 7 days |
| Rolling vs paper as printed: min %, windows positive | -1.75, 518 of 542 (96%) | -1.26, 515 of 542 (95%) | not committed |

Reading: the 15- and 30-minute results are the same story on all three indices: a 4% to 8% gain,
no regime in which the hybrid loses, a log-scale slope of 1 for the hybrid against 0.92 to 0.99 for
mcsGARCH, significance under every loss that is robust to proxy noise and not under MSE-type
losses, SPA p = 0.000 against the ladder, and not more than a one-week losing streak. Two
differences from Nifty: Patton's b = -1 loss is not significant on Fin Nifty at 15 and 30 minutes
(p 0.12 and 0.08; the b = -1 loss is already heavy-tail sensitive and Fin Nifty's sample is the
shortest), and the Bank Nifty hybrid at 15 minutes is beaten by its own GEX and implied-volatility
variants by a margin the SPA test calls significant. The last hour is the horizon that does not
travel: strong on Bank Nifty, absent on Fin Nifty (where the hybrid beats mcsGARCH on 56% of days
but loses badly on a few expiry days). Fin Nifty's own expiry days are also where the gain over the
paper as printed turns slightly negative at both horizons. Status: VERIFIED for 15 and 30 minutes
on both indices; last hour VERIFIED on Bank Nifty and FAILED on Fin Nifty.

## Commands and seeds

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/backtest_extended.py \
    --run results/pipeline_nse_v2 --p116 results/p116_benchmark_as_printed \
    --out results/backtest_extended --draws 5000 --block 10 --seed 9700001
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/backtest_extended.py --index banknifty   # results/backtest_extended/banknifty/
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/backtest_extended.py --index finnifty    # results/backtest_extended/finnifty/
PYTHONPATH=src python -m pytest tests/test_backtest_extended.py -q      # 9 tests, under 2 seconds
```

Run time 12 seconds on one core for Nifty (15 and 7 seconds for Bank Nifty and Fin Nifty) (`results/backtest_extended/run_info.json`: numpy 2.5.4, pandas
3.1.0, finished 2026-11-04). Only item 4 uses random numbers (seed 9700001; the stationary bootstrap
is `stationary_bootstrap_indices` in the script). The DM test is `niftygex.evaluation.diebold_mariano`
unchanged; the GW functions are imported from `experiments/referee_forecast_gw.py` unchanged.

## Limitations

* Everything is computed from the committed forecasts. Whether a forecast used only information
  available before its window cannot be seen in a csv. The look-ahead found in the daily EGARCH
  selection of the mcsGARCH code path (`reports/BACKTEST_STRESS_REPORT.md`, section A2, EGARCH versus
  EWMA) is untouched by this document and applies to every model here that contains mcsGARCH.
* The paper-as-printed model enters only through its daily QLIKE losses, so it is absent from the
  calibration, loss-function and horizon items.
* The regime cells are many (34 cells times three horizons times two benchmarks); the p-values in
  the cell tables are not adjusted for that. The GW slope tests are the right summary: no
  conditioning variable predicts the gain.
* The conditional cells use the previous day's values so that each cut is known in advance. A cut on
  the same day's VIX or realized variance would describe where the gain happened but could not be
  acted on.
* The participant-rule gamma flip exists on only 355 of the 792 test days (the panel has it for 397
  of 942 days), so those two cells are small.
* The Mincer-Zarnowitz regressions pool anchors within a day; the HAC covariance sums scores within
  the day and uses Newey-West across days, which is a reasonable but not the only way to treat the
  within-day dependence.
* The SPA p-values of 0.000 mean none of the 5,000 draws exceeded the statistic; the resolution is
  1 / 5,000.
* Mean block 10 days for the stationary bootstrap follows the earlier report; the block length was
  not varied.

## Status per item

| Item | Status | Where |
| --- | --- | --- |
| 1 regime cells and GW | VERIFIED | `regime_gains.csv`, `gw_tests.csv`, `gw_slope_coefficients.csv` |
| 2 calibration and encompassing | PARTIALLY VERIFIED (log scale well calibrated and encompassing holds; variance scale over-dispersed, outlier-driven) | `calibration_mincer_zarnowitz.csv`, `encompassing.csv`, `ratio_y_over_f.csv` |
| 3 alternative losses | VERIFIED under QLIKE, log MSE, volatility QLIKE, Patton b = -1; FAILED (not significant) under MSE and Patton b = 1 | `alternative_losses.csv` |
| 4 SPA | VERIFIED | `spa.csv`, `spa_per_model.csv` |
| 5 horizon consistency | VERIFIED | `horizon_consistency.csv` |
| 6 rolling gains | VERIFIED | `rolling_gain_summary.csv`, `rolling_gain_series.csv` |
| replication on Bank Nifty and Fin Nifty | VERIFIED at 15 and 30 minutes on both; last hour VERIFIED on Bank Nifty, FAILED on Fin Nifty | `results/backtest_extended/{banknifty,finnifty}/` |
| unit tests | VERIFIED (9 passed) | `tests/test_backtest_extended.py` |
