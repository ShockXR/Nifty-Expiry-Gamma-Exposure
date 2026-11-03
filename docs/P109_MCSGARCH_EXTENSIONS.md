# P109: what does the hybrid's correction fix in mcsGARCH?

Study P109, run by an AI agent (Claude) working for Shreyan Thanki on 3 November 2026.
**All numbers below are REAL DATA** (Nifty, Bank Nifty and Fin Nifty, NSE one-minute index data as in the
headline runs) unless marked SYNTHETIC. Every number comes from a file in `results/p109_mcsgarch_extensions/`
(file named next to each table) or, where marked, from an earlier committed results file.

* Registered plan: `docs/prereg/P109_MCSGARCH_EXTENSIONS.md`, commit `ceec2ec2d4287c1fa0a2616dfe8eac137fb2c3dd`
  (committed 08:49:56 UTC and pushed about 08:50 UTC, before any forecast of an extended model was made on real data
  and before any loss of this study was computed).
* Code (all in the plan commit, unchanged since): `experiments/p109_mcsgarch_extensions.py`; new code after the
  existing code in `src/niftygex/volatility.py` (`har_daily_variance`, `ext_q2`, `fit_garch_ext`, `McsGarchExt`).
  Tests (SYNTHETIC only): `tests/test_p109_mcsgarch_extensions.py`.
* Every p-value of the study: `results/p109_mcsgarch_extensions/tests.csv` (100 rows: 2 primary, 26 secondary,
  6 robustness, 66 exploratory).

*Note added 3 November 2026 after study P116:* here "mcsGARCH", "the anchor paper's model" and "the committed
mcsGARCH" mean our implementation (a separate time-of-day curve for expiry days, a 5-minute smoothing of the curves,
a zero mean), not the paper's own specification; the five extensions were added to it. Against the paper's
specification (one curve, no smoothing, ARMA(1,1) mean) see `docs/P116_BENCHMARK_AS_PRINTED.md`; there P109's
combination is 5.48% and 7.25% better on Nifty at 15 and 30 minutes (6.83% and 8.25% against our implementation),
and the hybrid 4.43% and 6.64%. The gains and shares below are measured against our implementation and were not
recomputed in P116.

Units. A **gain** is in points of mcsGARCH's mean daily QLIKE: 100 x (L_mcsGARCH - L_model) / L_mcsGARCH, so
positive means better than mcsGARCH. The hybrid's gain on Nifty is 5.80 points at 15 minutes and 7.65 points at
30 minutes. A **share** is a model's gain divided by the hybrid's gain on the same days: 100% means the model
recovers all of the hybrid's gain, more than 100% means it beats the hybrid.

## 1. The answer in plain words (REAL DATA)

Our hybrid forecasts how much the Nifty will move in the next 15 or 30 minutes. It starts from the anchor
paper's model (mcsGARCH) and multiplies its forecast by a correction learnt by LightGBM. Study P106 showed that
the correction gets all of its gain from mcsGARCH's own kind of information (past one-minute returns, the time of
day and the expiry-day flag). So the correction fixes something that mcsGARCH itself gets wrong. This study asked
**what**. Before seeing any result, we wrote down five small changes to mcsGARCH, each fixing one suspected
mistake, and one model with all five together (the **combination**).

**Does a readable model recover the hybrid's gain? Yes: most of it for sure, and all of it in the point
estimate, on these days.**

* The combination is **6.83 points better than mcsGARCH at 15 minutes and 8.25 points better at 30 minutes**
  (one-sided Diebold-Mariano p below 0.001 at both, Holm-adjusted below 0.001).
* The hybrid is 5.80 and 7.65 points better. So the combination recovers **118% and 108% of the hybrid's gain**
  (95% bootstrap intervals 85% to 167% and 80% to 132%).
* The registered verdict is **"supported: recovers most of the gain"**: both tests pass and the lower end of
  both intervals is above one half.
* **The hybrid does not beat the combination.** The combination is slightly better (by 1.03 and 0.60 points),
  but the difference is not significant (p 0.823 and 0.729 for "the hybrid is better"). The two cannot be told
  apart on these 792 days. This is against what we expected in the plan.

**What does the correction fix? Three things, and a fourth only together with the others.**

1. **The day's level so far (X3).** mcsGARCH's minute-by-minute part remembers only the last 20 minutes or so.
   Feeding it today's average squared return so far recovers **46% and 48%** of the hybrid's gain on its own
   (Holm p below 0.001 at both horizons).
2. **The time of day (X5).** mcsGARCH's average level is off at some times of day. One correction factor per
   forecast time, learnt on the training days, recovers **49% and 39%** on its own (Holm p below 0.001 and 0.003).
   The factors say that mcsGARCH's forecast is on average too low at 09:15 and in the 15:00 to 15:15 window, and
   too high in the 14:45 to 15:00 window.
3. **The overnight break (X2).** mcsGARCH runs its minute-by-minute part straight from yesterday's last minute
   into today's first. Letting it fade overnight recovers **12% and 16%** on its own (significant at 30 minutes,
   Holm p 0.015; not at 15 minutes, Holm p 0.181). Its gain is at the open (09:15 and 09:30); summed over all
   forecasts after 09:15 it is about zero. In the combination, only about 9% of how far the minute-by-minute
   part stood above or below its average at yesterday's close survives the night.
4. **The daily level (X1).** Replacing mcsGARCH's daily EGARCH by a HAR model of daily realized variance helps
   little on its own (19% and 3%, not significant). But inside the combination it matters: taking it out costs
   1.84 and 1.97 points (exploratory, raw p 0.035 and 0.051), mostly at the 09:15 forecast. Once yesterday's
   close no longer drives the open (X2), the open's forecast rests on the daily level, and a better daily level
   pays.
5. **Falls versus rises (X4)** does not help: alone it is slightly worse than mcsGARCH (-0.27 and -0.39 points,
   not significant), although the fitted leverage term is clearly positive.

**Where in the day.** At the 09:15 forecast the combination is 28.7 and 34.4 points better than mcsGARCH,
against 13.3 and 23.9 for the hybrid. At all the other forecast times together it recovers **93%** of the
hybrid's gain at both horizons (intervals 67% to 129% and 58% to 122%). So the result does not rest on the open
alone, though the open is where the combination does better than the hybrid.

**Where it fails: the last hour.** For the 14:30 forecast of the last hour, the combination is only 0.80 points
better than mcsGARCH (p 0.416), 13% of the hybrid's 6.36 points. The hybrid's edge over the combination there
(5.56 points) is not significant either (p 0.064, Holm 0.129).

**Bank Nifty and Fin Nifty, nothing retuned.** The combination beats mcsGARCH at both horizons on both indices
(p below 0.001) and recovers **127% and 136%** (Bank Nifty) and **97% and 103%** (Fin Nifty) of the hybrid's
gain. The hybrid does not beat it on either index. There the overnight fade (X2) is a larger piece and the day's
level (X3) a smaller one.

**What it means.** On these days, most or all of what the LightGBM correction adds to mcsGARCH can be written as
three or four plain fixes to mcsGARCH itself. That is a clear, describable answer for the paper. It also means
that the hybrid's 5.80 and 7.65 points are a gain over our implementation of mcsGARCH, not over a carefully fixed
mcsGARCH (wording corrected after P116: against the paper's own specification the combination is 5.48% and 7.25%
better at 15 and 30 minutes, and the hybrid 4.43% and 6.64%). The combination was designed after P106 on the same days, so it needs new data before it can be a
headline (Section 15).

## 2. What was registered (in short)

* **Models.** The committed mcsGARCH (daily AR(1)-EGARCH(1,1) level, diurnal curve with a separate expiry-day
  curve, intraday Student-t GARCH(1,1)) and five extensions, each switching on one change:

  | Name | Suspected mistake | Change to mcsGARCH |
  |---|---|---|
  | X1 `x1_har` | the daily level is stale and noisy | daily level from a log-HAR of yesterday's, last week's and last month's realized variance (OLS on the training days) |
  | X2 `x2_overnight` | yesterday's close leaks into today's open | at the first minute of each day the intraday part is pulled toward its average: q_open = qbar + delta (q_carry - qbar), delta estimated |
  | X3 `x3_todayrv` | the intraday part forgets the day's level too fast | GARCH-X term gamma_x times today's running mean of normalised squared returns |
  | X4 `x4_leverage` | falls and rises move volatility alike | GJR term gamma_l x^2 on falling minutes |
  | X5 `x5_calibration` | the average level is wrong at some times of day | forecast at each anchor minute times c_i, the mean of y / forecast on the training days |

  The **combination** has X1 to X4 fitted jointly by maximum likelihood, then X5's calibration. Five
  exploratory leave-one-out variants (`comb_no_x1` to `comb_no_x5`) drop one extension each (Nifty only).
* **Design.** The committed walk-forward: the first 150 days for training only, then a refit on all earlier days
  before every block of 21 trading days. Nifty 792 test days (10 August 2023 to 2 November 2026), Bank Nifty 660,
  Fin Nifty 604. The hybrid's committed forecasts (seed set 7) are used as they are.
* **Primary test** (2 rows, Holm over both): the combination has lower mean daily QLIKE than mcsGARCH at 15 and
  30 minutes (one-sided Diebold-Mariano with Newey-West variance, HLN factor, Student t). With it, the share of
  the hybrid's gain recovered and its 95% stationary-bootstrap interval (mean block 10 days, 10,000 draws).
* **Decision rule.** "Supported, recovers most of the gain" if both primary Holm p are below 0.05 and the lower
  end of both share intervals is at least 0.5. "Supported in the point estimate only" if the shares are at least
  0.5 but an interval reaches below. "Contradicted" if the upper end of both intervals is below 0.5.
  "Inconclusive" otherwise.
* **Secondary families** (Holm within each): S1 each extension alone (10 rows); S2 the hybrid against the
  combination (2); S3 the last hour (2); S4 the 09:15 anchor and the other anchors (4); S5 Bank Nifty and S6 Fin
  Nifty (4 each). **Robustness**: R1 without the 42 EWMA-fallback days, R2 before the closing auction, R3 S2 with
  the seed-averaged hybrid. **Exploratory** (no adjustment): E1 leave-one-out, E2 each extension at 09:15 and at
  the other anchors, E3 expiry days, E4 each extension in the last hour, E5 P106's `mcsinfo` hybrid against the
  combination, E6 each extension on Bank Nifty and Fin Nifty.

## 3. The primary result (REAL DATA, Nifty, 792 test days)

From `verdict.csv`, `tests.csv`, `gains.csv` and `shares.csv`.

| Model | 15 min: gain (points) | share of the hybrid's gain [95% interval] | p (Holm) | 30 min: gain | share [95% interval] | p (Holm) |
|---|---|---|---|---|---|---|
| hybrid (committed, seed set 7) | +5.80 | 100% | | +7.65 | 100% | |
| **combination** | **+6.83** | **118% [85%, 167%]** | **below 0.001 (below 0.001)** | **+8.25** | **108% [80%, 132%]** | **below 0.001 (below 0.001)** |
| X1 HAR daily level | +1.12 | 19% [-11%, 58%] | 0.124 (0.496) | +0.21 | 3% [-29%, 35%] | 0.434 (1.000) |
| X2 overnight fade | +0.68 | 12% [-1%, 29%] | 0.036 (0.181) | +1.22 | 16% [4%, 31%] | 0.003 (0.015) |
| X3 today's level | +2.69 | 46% [27%, 72%] | below 0.001 (below 0.001) | +3.66 | 48% [28%, 65%] | below 0.001 (below 0.001) |
| X4 leverage | -0.27 | -5% [-19%, 5%] | 0.814 (1.000) | -0.39 | -5% [-15%, 3%] | 0.868 (1.000) |
| X5 time-of-day factor | +2.83 | 49% [33%, 67%] | below 0.001 (below 0.001) | +2.95 | 39% [20%, 55%] | below 0.001 (0.003) |
| P106's `mcsinfo` hybrid (for reference) | +6.06 | 104% [94%, 118%] | | +8.01 | 105% [93%, 116%] | |

The p-values test "better than mcsGARCH" (one-sided). The combination row is the primary family (P1); the five
extension rows are family S1 (Holm over 10 rows).

* **Verdict** (`verdict.csv`): "supported: recovers most of the gain (interval above one half at both horizons)".
  Share 1.178 (interval 0.846 to 1.675) at 15 minutes and 1.079 (0.800 to 1.325) at 30 minutes.
* **S2, the hybrid against the combination**: the hybrid's gain minus the combination's is -1.03 points at 15
  minutes (DM statistic -0.93, p 0.823, Holm 1.000) and -0.60 points at 30 minutes (-0.61, p 0.729, Holm 1.000).
  The test asks whether the hybrid is better; it is not, and the point estimate goes the other way.
* **Adding up.** The five single gains add to 7.05 points at 15 minutes (combination 6.83) and 7.65 points at 30
  minutes (combination 8.25), computed from `shares.csv`. So on Nifty the pieces roughly add up: there is no
  large overlap and no large extra from combining them. On Bank Nifty they overlap more (singles add to 6.49 and
  8.53 points, combination 4.97 and 6.83).

## 4. The five extensions one by one (REAL DATA, Nifty)

**Alone** (S1, table above): X3 and X5 each fix a misspecification that matters at both horizons (Holm p below
0.05 at both). X2 does so at 30 minutes only. X1 and X4 do not.

**Leave one out of the combination** (E1, exploratory, no adjustment; `tests.csv`). The rise in the
combination's loss when one extension is removed, points of mcsGARCH's QLIKE:

| Removed | 15 min: rise | raw p | 30 min: rise | raw p | last hour: rise | raw p |
|---|---|---|---|---|---|---|
| X1 HAR daily level | +1.84 | 0.035 | +1.97 | 0.051 | -2.12 | 0.866 |
| X2 overnight fade | +1.27 | below 0.001 | +1.87 | below 0.001 | -0.87 | 0.821 |
| X3 today's level | +1.03 | 0.005 | +2.04 | below 0.001 | +1.16 | 0.260 |
| X4 leverage | +0.15 | 0.179 | +0.16 | 0.159 | -0.23 | 0.661 |
| X5 time-of-day factor | +1.68 | below 0.001 | +1.72 | 0.012 | +1.77 | 0.110 |

* Every extension except X4 still adds to the combination at 15 and 30 minutes. X4 adds nothing that can be
  told from zero.
* **X1 is worth more inside the combination than alone** (removing it costs 1.84 and 1.97 points; alone it gains
  1.12 and 0.21). Most of the difference is at 09:15: without X1 the combination's 09:15 gain falls from 28.69 to 14.10
  points at 15 minutes and from 34.37 to 17.74 at 30 minutes (`gains.csv`, "09:15 anchor only").
* **In the last hour no removal makes the combination significantly worse**, and removing X1 or X2 makes it
  slightly better (not significant).

## 5. What the fitted models say, in plain words (REAL DATA, Nifty)

Means over the 38 refits, with the smallest and largest refit in brackets (`params_summary.csv`; every refit in
`params_by_refit.csv`).

| Parameter | mcsGARCH | X2 alone | X3 alone | X4 alone | combination |
|---|---|---|---|---|---|
| alpha (weight of the last minute's squared return) | 0.045 (0.025 to 0.052) | 0.042 (0.025 to 0.048) | 0.056 (0.034 to 0.062) | 0.033 (0.019 to 0.039) | 0.035 (0.023 to 0.041) |
| beta (memory of the intraday part) | 0.950 (0.943 to 0.975) | 0.954 (0.947 to 0.975) | 0.914 (0.901 to 0.947) | 0.950 (0.944 to 0.975) | 0.923 (0.912 to 0.937) |
| delta (share of yesterday's closing deviation from the average that survives the night) | 1 (fixed) | 0.42 (0.25 to 1.00) | 1 (fixed) | 1 (fixed) | 0.09 (0.00 to 0.23) |
| gamma_x (weight of today's running mean) | 0 | 0 | 0.024 (0.014 to 0.038) | 0 | 0.018 (0.013 to 0.024) |
| gamma_l (extra weight on a falling minute) | 0 | 0 | 0 | 0.023 (0.011 to 0.028) | 0.033 (0.028 to 0.043) |
| nu (Student t degrees of freedom) | 6.45 (5.87 to 8.45) | 6.59 (5.99 to 8.55) | 6.54 (5.92 to 8.77) | 6.49 (5.90 to 8.46) | 6.66 (6.03 to 8.96) |

* **The night.** In mcsGARCH, whatever the intraday part learnt in the last half hour of yesterday carries in full
  into 09:15 (delta = 1). With the overnight fade alone, 42% of that deviation survives the night on average. In the
  combination, where the daily level also comes from the HAR model, only **9%** survives, and in some refits none.
  On Bank Nifty and Fin Nifty the combination keeps 19% and 14% (means over 32 and 29 refits).
* **The day's level.** With today's running mean in the model, beta falls from 0.950 to 0.914 (alone) or 0.923
  (combination): the intraday part leans less on the last few minutes because it now has a steadier measure of
  the day's level.
* **Falls versus rises.** In the combination a falling minute's squared return gets weight alpha + gamma_l =
  0.035 + 0.033, nearly twice a rising minute's 0.035. This is a clear asymmetry in the fit, but it does not
  improve the forecasts (X4 alone -0.27 and -0.39 points; leaving it out of the combination changes the loss by
  +0.15 and +0.16 points).
* **The HAR daily level** (X1): log daily variance = b0 + 0.49 log(yesterday's realized variance) + 0.14 log(last
  week's) + 0.14 log(last month's), on average over refits (b_d 0.43 to 0.59, b_w 0.03 to 0.23, b_m 0.08 to 0.17).
* **The time-of-day factors** (X5, `calibration_factors.csv`, mean over refits, 15-minute target): c_i is the
  average of (what happened) / (forecast) at that time of day on the training days, so above 1 means the
  forecast was too low.

  | Forecast time | X5 alone (on mcsGARCH) | combination |
  |---|---|---|
  | 09:15 | 1.168 | 0.973 |
  | 09:30 | 1.102 | 1.091 |
  | 09:45 | 1.069 | 1.094 |
  | 14:45 | 0.830 | 0.831 |
  | 15:00 | 1.286 | 1.259 |
  | 15:15 | 1.083 | 1.063 |

  At 09:15, mcsGARCH's forecast is on average too low (1.17), but the combination's is about right (0.97): X1
  and X2 already fix the open. Around 15:00 both models have the same pattern: the 14:45 to 15:00 window is
  forecast too high and the 15:00 to 15:15 window too low. This study does not say why; the factors only say
  where. At 30 minutes the 09:15 factors are 1.184 and 0.990, and the 15:00 factors 1.250 and 1.218.
* **Fits.** Every fitted model converged: all 883 model rows of `params_by_refit.csv` (456 on Nifty, 224 on Bank
  Nifty, 203 on Fin Nifty; X5 and `comb_no_x5` reuse the fits of mcsGARCH and the combination). The second registered starting point (next to plain mcsGARCH) gave the better likelihood in 13 of
  the combination's 38 Nifty refits and the first in 25.

## 6. Where in the day (REAL DATA, Nifty)

**S4** (Holm over 4 rows; `tests.csv`, `shares.csv`):

| Anchors | 15 min: combination gain | p (Holm) | share of the hybrid's gain [95%] | hybrid's gain | 30 min: combination gain | p (Holm) | share [95%] | hybrid's gain |
|---|---|---|---|---|---|---|---|---|
| 09:15 only | +28.69 | 0.004 (0.004) | 216% [104%, 492%] | +13.30 | +34.37 | 0.002 (0.004) | 144% [102%, 173%] | +23.86 |
| all other anchors | +4.72 | below 0.001 (below 0.001) | 93% [67%, 129%] | +5.08 | +5.54 | below 0.001 (below 0.001) | 93% [58%, 122%] | +5.96 |

* The 09:15 forecast is 1 of 25 forecasts a day at 15 minutes, but it holds 37% of the combination's total QLIKE
  reduction (39% at 30 minutes), against 20% (29%) of the hybrid's (`gain_by_anchor.csv`).
* Away from the open the combination recovers 93% of the hybrid's gain at both horizons.

**Each extension at 09:15 and elsewhere** (E2, exploratory, raw p; `tests.csv`):

| Extension | 09:15, 15 min | 09:15, 30 min | other anchors, 15 min | other anchors, 30 min |
|---|---|---|---|---|
| X1 HAR daily level | +5.08 (0.301) | +6.07 (0.282) | +0.74 (0.076) | -0.40 (0.681) |
| X2 overnight fade | +10.35 (0.003) | +13.25 (below 0.001) | -0.25 (0.838) | -0.03 (0.545) |
| X3 today's level | +11.27 (0.005) | +12.11 (0.002) | +1.86 (below 0.001) | +2.79 (below 0.001) |
| X4 leverage | -3.55 (0.873) | -3.62 (0.861) | +0.04 (0.417) | -0.06 (0.601) |
| X5 time-of-day factor | +8.94 (0.012) | +10.16 (0.007) | +2.24 (below 0.001) | +2.20 (0.007) |

* X2's gain is at the open, as expected (at 09:30 too: -8.83% and -12.72% of mcsGARCH's QLIKE at that anchor,
  `gain_by_anchor.csv`); over the other anchors together it is about zero. X3 and X5 help both at the open and
  through the day.
* At the 09:15 anchor X3 probably helps because at the day boundary its running mean is yesterday's whole-day
  average, a steadier signal than yesterday's last half hour (not tested separately).

**Other anchors** (`gain_by_anchor.csv`, % change of mcsGARCH's QLIKE, negative is better): at 09:30 the hybrid
-14.11 and the combination -20.29 (15 minutes); at 15:00 the hybrid -12.51 and the combination -7.78. So the
hybrid keeps an edge at the 15:00 forecast of the 15-minute target (at 30 minutes: -14.79 and -15.37).

## 7. The last hour (REAL DATA, Nifty, 14:30 forecast)

S3 (Holm over 2 rows) and E4 (exploratory):

* The combination against mcsGARCH: **+0.80 points**, DM statistic 0.21, p 0.416 (Holm 0.416). Share of the
  hybrid's 6.36 points: 13% [-127%, 111%].
* The hybrid against the combination: +5.56 points, p 0.064 (Holm 0.129). Not significant.
* Each extension alone: X1 +1.10 (p 0.365), **X2 -5.16** (p 0.987 for "better"), X3 +2.93 (0.143), X4 +0.29
  (0.351), X5 +2.10 (0.046).
* P106's `mcsinfo` hybrid gains 5.07 points here (80% of the hybrid's gain, interval 52% to 115%).

The overnight fade hurts the last hour, although the 14:30 forecast has nothing to do with the night. A likely
reason, not tested here: X2 changes the other fitted parameters of the intraday part (its persistence and its
long-run level), and the 60-minute forecast leans on them more than the 15-minute one. The combination does not
explain what the hybrid does in the last hour; with one forecast a day the last hour is also the noisiest
target.

## 8. Bank Nifty and Fin Nifty (REAL DATA, nothing retuned)

S5 and S6 (Holm over 4 rows each) and E6 (exploratory): same code, same starting values, same rules.

| Index, target | hybrid's gain | combination's gain | p (Holm) | share [95%] | hybrid minus combination | p (Holm) |
|---|---|---|---|---|---|---|
| Bank Nifty, 15 min (660 days) | +3.92 | +4.97 | below 0.001 (below 0.001) | 127% [77%, 192%] | -1.05 | 0.841 (1.000) |
| Bank Nifty, 30 min | +5.03 | +6.83 | below 0.001 (below 0.001) | 136% [79%, 221%] | -1.80 | 0.867 (1.000) |
| Fin Nifty, 15 min (604 days) | +4.90 | +4.73 | below 0.001 (below 0.001) | 97% [58%, 166%] | +0.17 | 0.449 (0.899) |
| Fin Nifty, 30 min | +6.09 | +6.26 | below 0.001 (below 0.001) | 103% [61%, 173%] | -0.17 | 0.542 (0.899) |

Each extension alone (E6, gain in points, raw p for "better than mcsGARCH"):

| Extension | Bank Nifty 15 min | Bank Nifty 30 min | Fin Nifty 15 min | Fin Nifty 30 min |
|---|---|---|---|---|
| X1 HAR daily level | +1.24 (0.138) | +2.03 (0.118) | +1.08 (0.166) | +0.72 (0.337) |
| X2 overnight fade | +2.31 (below 0.001) | +3.18 (below 0.001) | +1.61 (below 0.001) | +2.32 (below 0.001) |
| X3 today's level | +1.14 (0.022) | +1.58 (0.037) | +0.93 (0.091) | +1.39 (0.114) |
| X4 leverage | -0.05 (0.689) | -0.19 (0.907) | -0.06 (0.695) | -0.30 (0.922) |
| X5 time-of-day factor | +1.86 (below 0.001) | +1.93 (below 0.001) | +1.83 (below 0.001) | +2.00 (below 0.001) |

* The main result replicates on both indices: the combination beats mcsGARCH and the hybrid does not beat the
  combination.
* The ranking of the pieces differs: on Bank Nifty and Fin Nifty the overnight fade (X2: 59% and 63%, 33% and 38%
  of the hybrid's gain) and the time-of-day factor (X5) lead, and today's level (X3) is smaller (29% and 31%,
  19% and 23%). X4 never helps.

## 9. Robustness and other exploratory results (REAL DATA, Nifty)

**Robustness** (Holm within each family; `tests.csv`, `shares.csv`):

| Row | 15 min | 30 min |
|---|---|---|
| R1 without the 42 EWMA-fallback days (750 days): combination's gain, p (Holm), share | +6.91, below 0.001 (below 0.001), 117% [82%, 167%] | +8.34, below 0.001 (below 0.001), 106% [78%, 131%] |
| R2 before the closing auction (729 days) | +6.20, below 0.001 (below 0.001), 112% [77%, 161%] | +6.84, below 0.001 (below 0.001), 100% [68%, 129%] |
| R3 the seed-averaged hybrid (21 seed sets) against the combination: hybrid minus combination, p (Holm) | -1.06, 0.829 (1.000) | -0.34, 0.631 (1.000) |

The result does not depend on the EWMA-fallback days, the closing auction or the hybrid's seed.

**Expiry days** (E3, 169 days): the combination is +7.33 and +7.64 points better than mcsGARCH (p 0.003 and
0.005), 110% [70%, 159%] and 123% [62%, 175%] of the hybrid's gain; the hybrid minus the combination is -0.69 and
-1.41 points (p 0.687 and 0.782).

**P106's `mcsinfo` hybrid against the combination** (E5): `mcsinfo`'s gain minus the combination's is -0.77
and -0.25 points (p 0.747 and 0.594 for "`mcsinfo` is better"). The combination is level with the best hybrid
P106 found.

**By expiry regime** (descriptive, no test; `shares.csv`):

| Regime | days | 15 min: combination's gain, share [95%] (hybrid's gain) | 30 min: combination's gain, share [95%] (hybrid's gain) |
|---|---|---|---|
| several weekly expiries (to November 2024) | 312 | +7.26, 140% [67%, 267%] (+5.19) | +7.94, 118% [55%, 167%] (+6.70) |
| one weekly expiry, Thursday | 192 | +3.79, 73% [28%, 114%] (+5.16) | +3.89, 61% [13%, 105%] (+6.42) |
| one weekly expiry, Tuesday | 288 | +8.66, 127% [86%, 213%] (+6.83) | +12.27, 126% [100%, 148%] (+9.73) |

In the Thursday regime the combination falls short of the hybrid, and X1 alone is worse than mcsGARCH (-1.68 and
-3.20 points). In the last hour of the Thursday regime the hybrid's gain is -0.04 points, so shares there are
meaningless (they are in `shares.csv` but should not be read).

## 10. What we expected, and what happened (REAL DATA)

| Registered expectation | Outcome |
|---|---|
| the combination beats mcsGARCH at both horizons | **as expected** (p below 0.001 at both) |
| it recovers between one third and two thirds of the hybrid's gain | **not as expected: more**, 118% and 108% (intervals 85% to 167%, 80% to 132%) |
| X1 and X2 are the largest single extensions, with most of their gain at 09:15 and 09:30 | **not as expected**: X3 and X5 are the largest; X1 alone is not significant; X2 is the third, and its gain is indeed all at the open |
| X5 gives a smaller gain spread over the day | **partly**: spread over the day, yes, but it is one of the two largest (49% and 39%) |
| X3 and X4 are small | **X4 as expected** (slightly negative); **X3 not**: 46% and 48% |
| the hybrid still beats the combination (S2) | **not as expected**: the combination is slightly better, not significantly |

## 11. Checks (REAL DATA unless marked SYNTHETIC)

* **Reproduction of the committed forecasts** (`reproduction_check.csv`). Through this study's driver, the rebuilt
  mcsGARCH forecasts and targets y are identical to the last bit to the committed ones on every row: Nifty 19,800
  (15 minutes), 19,008 (30 minutes) and 792 (last hour); Bank Nifty 16,500 and 15,840; Fin Nifty 15,100 and
  14,496. `analyze` refuses to run otherwise.
* **Reproduction of P106 and P99.** This study's daily QLIKE of mcsGARCH equals P106's committed daily losses,
  and its hybrid daily QLIKE equals P99's seed-set-7 daily losses, on all 792 days for each Nifty target (largest
  relative difference 0).
* **Inputs.** Every raw file matches `data/MANIFEST.csv` (`run_info.json`, `input_sha256.<index>.manifest_check`):
  1,891 files for Nifty, 1,891 for Bank Nifty and 1,711 for Fin Nifty, none differing.
* **No look-ahead**, by construction and by test: the HAR daily level for day t uses realized variance up to day
  t-1, with coefficients from training days only (test `test_har_daily_variance_ols_and_no_lookahead`); the
  09:15 forecast uses only yesterday's carry and yesterday's mean; within a day the running mean uses minutes
  before the forecast time; the calibration factors use training days only (test `test_calibration_factors_definition_and_optimality`); the
  diurnal curve and every parameter are fitted on the days before the block. With every switch off, the new code
  reproduces mcsGARCH bit for bit (above and in a synthetic test). The multi-step forecast was checked against a
  closed form and a Monte Carlo of the full model (SYNTHETIC).
* **Fits.** All 883 model rows converged. In the first Nifty refit (150 training days), X2, X3 and X4 alone and
  `comb_no_x1` reached the upper bound of persistence (0.99999), as plain mcsGARCH's own GARCH does in some
  refits; in those fits delta is 1, so the long-run level this implies plays no role at the open. One warning
  (overflow in the exponential that maps the Student t degrees of freedom) appeared once in the Fin Nifty run
  while the optimiser searched; every fitted value is finite.
* **Tests.** `tests/test_p109_mcsgarch_extensions.py` (SYNTHETIC only, 13 tests, committed with the plan) passes.
  The full suite (`python -m pytest -q tests/`), run at 09:11 to 09:15 UTC before the phase-2 commit: 627 tests,
  620 passed, 7 skipped, none failed. The 7 skipped tests need the raw Bank Nifty and Fin Nifty files, which were
  unlinked from the working copy before committing; run again with the raw data in place
  (`tests/test_symbol_banknifty.py`, `tests/test_symbol_finnifty.py`), all 19 tests of those two files passed.

## 12. Run time

* **Forecasts** (stage 1, 2 worker processes), on 3 November 2026 (`run_stdout_*.txt`, `run_info.json`,
  `stage1`):
  * Nifty 7.3 minutes (08:50 to 08:57 UTC, including 64 s to load the market data; 38 refits, 10 distinct fits
    each);
  * Bank Nifty 3.1 minutes and Fin Nifty 3.0 minutes (finished 09:04 UTC);
  * the jobs' own times add up to 728.2, 264.8 and 236.3 seconds (`timing.csv`).
* **Analysis** (stage 2): 87.3 seconds (`run_info.json`, `seconds_stage2`).
* The plan expected 15 to 30 minutes for Nifty and 10 to 20 for each other index. The whole run took about 16
  minutes.
* **Full test suite** before the phase-2 commit: about 4 minutes, plus 71 seconds for the raw-data tests.

## 13. Deviations from the plan

1. **None in the design.** Every registered model, test, family and output was run as written, on all three
   indices, with no change to the code after the plan commit (`analysis_commit` in `run_info.json` is the plan
   commit, and the phase-2 commit changes no code). Bank Nifty and Fin Nifty finished at 09:04 UTC, long before
   the 17:00 UTC cut-off, so nothing was dropped. No run was restarted; no block was recomputed.
2. **Commands.** The plan's commands set `PYTHONPATH=src`; the run used the absolute path of the same `src`
   folder (a sandbox rule). `analyze` was given `--plan-commit ceec2ec2d4287c1fa0a2616dfe8eac137fb2c3dd`. Same
   code.
3. **The sum of the five single shares** is listed in the plan among the things reported. `analyze` does not write
   it as its own row; it is computed in Section 3 from `shares.csv` (point estimates only, no interval).
4. **Reading notes, not deviations.** Shares in the last hour of the Thursday regime have a denominator near zero
   (the hybrid's gain is -0.04 points) and are not read. The E5 estimate is `mcsinfo`'s gain minus the
   combination's (the plan did not state its sign).
5. **Tests.** The synthetic tests were added with the plan (phase 1); none was added in phase 2.
6. **Full test suite.** Run before the phase-2 commit (Section 11): 620 passed and 7 skipped for lack of raw
   data, then those 7 passed with the raw data in place. None failed.

## 14. Caveats

1. **Same days, chosen after P106.** The five extensions and the combination were chosen knowing P106's
   attribution, the anchor results and the feature importances, all from these 792 test days. This study explains
   the hybrid's gain on those days; it is not an independent confirmation. Bank Nifty and Fin Nifty are less
   exposed (nothing was tuned there), but P106's results on them were also known.
2. **The time-of-day factor (X5) is fitted in-sample** on the training days, like the hybrid's correction, and is
   the least structural of the five. It says where mcsGARCH's level is off, not why.
3. **The hybrid and the combination cannot be told apart.** "The combination beats the hybrid" is not
   established: the point estimate favours it, the tests do not.
4. **The intervals are wide.** The share interval at 15 minutes runs from 85% to 167%. "Most of the gain" is
   firm; "all of it" is the point estimate.
5. **The open matters a lot.** About 37% to 39% of the combination's QLIKE reduction is at 09:15. Away from the
   open it recovers 93% (intervals 67% to 129% and 58% to 122%), so the conclusion holds there too, less
   precisely.
6. **The last hour is not explained.** The combination gains 0.80 points there against the hybrid's 6.36.
7. **The Thursday regime is weaker** (73% and 61%, intervals reaching 28% and 13%). Regime results are
   descriptive.
8. **Five extensions are not all possible ones**, and the extended likelihood was maximised from two starting
   points only. The hybrid is one seed set (R3 repeats S2 with the 21-seed average and finds the same). The
   leverage term's multi-step forecast assumes symmetric errors.

## 15. What this changes in the project's claims

* **We can now say in plain words what the hybrid's correction does.** On Nifty it mostly (a) feeds mcsGARCH the
  day's realized level so far, (b) corrects mcsGARCH's average level at some times of day, and (c) at the open,
  removes yesterday's closing volatility from today's first forecast and uses a better daily level instead.
  Leverage plays no part. The same pieces work on Bank Nifty and Fin Nifty, with more weight on the open.
* **The hybrid's gain should be described as a gain over our implementation of mcsGARCH, not over the best
  econometric model.** (Wording corrected after P116, which measures the gain against the paper's own
  specification: 4.43% and 6.64% on Nifty at 15 and 30 minutes.) A parametric mcsGARCH with five pre-specified fixes reaches the same loss on these days (Nifty, Bank
  Nifty and Fin Nifty), and the hybrid does not beat it at any registered horizon. Any sentence that presents
  the 5.80 and 7.65 points as something only machine learning could find should be softened.
* **The hybrid may keep an edge in the last hour** (5.56 points over the combination, not significant), and at
  the 15:00 forecast of the 15-minute target. These are leads, not results.
* **The combination is a new model chosen after seeing these data.** It should not replace mcsGARCH as the
  headline benchmark, or be reported as beating it, without its own registered test on data not used here (for
  example the months after 2 November 2026, or another index chosen in advance). It can be reported now as an
  explanation of the hybrid's gain, with the caveats above.

## 16. Files

All in `results/p109_mcsgarch_extensions/` (REAL DATA):

* `tests.csv`: every p-value (100 rows), with the registered columns; `tests_detail.csv` adds identifiers and
  sample details.
* `verdict.csv`: the primary verdict and the two shares with their intervals.
* `extensions.csv`: one row per extension (gain, share, Holm p, leave-one-out rise).
* `gains.csv`, `shares.csv`, `gain_by_anchor.csv`: the tables above, by index, target and subset.
* `params_by_refit.csv`, `params_summary.csv`, `calibration_factors.csv`: every refit's parameters and
  calibration factors, and their summaries.
* `daily_losses_<index>_<target>.csv`: each model's daily QLIKE (with the hybrid's, `mcsinfo`'s and the
  seed-averaged hybrid's for Nifty), from which every number can be recomputed.
* `reproduction_check.csv`, `timing.csv`, `run_info.json`: checks, timings, plan commit, settings, input hashes
  and versions.
* `run_stdout_nifty.txt`, `run_stdout_banknifty.txt`, `run_stdout_finnifty.txt`, `analyze_stdout.txt`: logs.

The forecasts of every model are in `data/processed/p109/` (git-ignored, about 27 MB).

## 17. How to reproduce

From the repository root, with the raw data listed in `data/MANIFEST.csv` in `data/raw`:

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py run --index NIFTY --workers 2 \
    > results/p109_mcsgarch_extensions/run_stdout_nifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py run --index BANKNIFTY --workers 2 \
    > results/p109_mcsgarch_extensions/run_stdout_banknifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py run --index FINNIFTY --workers 2 \
    > results/p109_mcsgarch_extensions/run_stdout_finnifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p109_mcsgarch_extensions.py analyze \
    --plan-commit ceec2ec2d4287c1fa0a2616dfe8eac137fb2c3dd > results/p109_mcsgarch_extensions/analyze_stdout.txt 2>&1
```

About 16 minutes on 2 cores. The synthetic tests: `PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q
tests/test_p109_mcsgarch_extensions.py`. A synthetic end-to-end run of all stages: add `--synthetic` (labelled
SYNTHETIC).
