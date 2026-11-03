# Model iterations on real data

**REAL DATA.** Everything in this document comes from real Indian market data: NSE's F&O bhavcopies (every listed Nifty option and future, settlement prices, open interest, volume), NSE participant-wise open interest, India VIX and NSE's daily index values, plus third-party 1-minute Nifty 50 bars that were checked day by day against NSE's official values. Sample: 942 trading days, 2 January 2023 to 2 November 2026. No number here comes from simulated data, except where a sentence says "synthetic" explicitly.

Written for a reader who is new to volatility forecasting. Section 1 gives the findings. Section 2 explains how to read the numbers. Section 3 walks through each iteration: what changed, why, and what happened, including what did not work. Section 4 is the final scoreboard and the GEX ablation. Section 5 lists the limitations.

## 1. Findings

All numbers are REAL DATA, out of sample, from `results/pipeline_nse_v2/` (792 test days from 10 August 2023 to 2 November 2026, 169 of them Nifty expiry days). "Better" means lower QLIKE loss than the paper's mcsGARCH, refitted every 21 trading days like every other model. p-values are one-sided Diebold-Mariano tests on daily average losses.

1. **The paper's model can be beaten at every horizon, and on expiry days too.** The hybrid (mcsGARCH multiplied by a LightGBM correction, new base features, no GEX) is better by 5.8% (next 15 minutes), 7.6% (next 30 minutes) and 6.4% (last hour) on all test days, all with p below 0.001. On expiry days it is better by 6.6% (p 0.001), 6.2% (p below 0.001) and 7.9% (p 0.010). Its daily loss is lower than mcsGARCH's on 71% and 74% of days at 15 and 30 minutes, but only on 54% of days for the last hour.
2. **It holds before the closing auction and in the hold-out.** On the 729 test days before NSE's closing auction began (3 August 2026), the gains are 5.5%, 6.9% and 7.0% (expiry days 5.4%, 5.4% and 6.5%; all p at most 0.036). On the hold-out (1 July 2025 onwards, evaluated once) they are 6.4%, 9.2% and 7.1%, against 5.3%, 6.6% and 5.3% on the development window where the design was chosen.
3. **A good part of the gain is at the open, but not all of it.** At the 09:15 forecast (which must absorb the overnight news) the hybrid is 13.3% and 23.9% better; at all later anchors it is still 5.1% and 6.0% better (p below 0.001). With each model's worst 1% of losses removed it is 5.6%, 6.7% and 6.4% better. It is better in all three expiry regimes at 15 and 30 minutes. Plain LightGBM is not: it is no better than mcsGARCH in the first regime (to November 2024; +1.4% and +4.8%) and much better afterwards.
4. **MSE tells a different story.** Under mean squared error the paper's model is about 6.7 times worse than a simple HAR at 15 minutes, because of a handful of opening forecasts: its ten largest squared errors are 97% of its total, the worst being 09:15 on 5 June 2024 (the morning after the election-result crash), when it forecast twenty times the realized variance. The hybrid inherits part of this. None of the MSE gaps against mcsGARCH at 15 and 30 minutes is statistically significant (p 0.13 to 0.18): they rest on a few windows. Taking mcsGARCH's daily level from the option straddle instead of from an EGARCH on past returns removes most of the problem (MSE 76% to 78% lower than mcsGARCH at 15 and 30 minutes).
5. **GEX does not help, under any sign rule.** Adding gamma exposure to the hybrid changes its loss by -0.42% to +0.14% at 15 and 30 minutes and by +0.9% to +1.6% in the last hour on all days. On all days, on expiry days and before the auction, no cell survives the Holm adjustment for multiple tests. The sign rule (US convention, Pro participant, client mirror) does not matter either: in the hybrid, signed GEX is never significantly better than unsigned gross gamma after Holm. The only cells that pass Holm at 5% are at 30 minutes in the hold-out (gross -0.98%, US sign -1.10%, Pro sign -0.63%), but there the placebo, GEX of a random earlier day, also "helps" (-0.5%, and -2.0% in the auction period), and the same comparison on the development window shows nothing. I read this as noise from a period of structural change, not as information.
6. **What helped:** refitting mcsGARCH every 21 days (it is 1.6% to 2.1% better than with the old 63-day schedule, so the benchmark got stronger); option-implied features for the linear model at 15 and 30 minutes (ridge log-HAR, 1.3 to 2.0 points better with them, p 0.003 to 0.025; not in the last hour) and as mcsGARCH's daily level in the first regime (5% to 6% better before November 2024); training the last-hour models on every 60-minute window of the day; averaging LightGBM over 5 seeds; and tuning plain LightGBM inside the training window (5.1%, 5.9% and 7.2% better than mcsGARCH).
7. **What did not help:** the implied block for the tree hybrid (changes of -0.4 to +0.2 points, not significant); a hybrid on the implied-level mcsGARCH (`hybridiv`, no better than `hybrid`, and worse in the last hour); tuning the hybrids; the straddle alone as a forecast (26% to 48% worse than mcsGARCH); and every GEX set.

## 2. How to read the numbers

**The task.** At a given minute of the trading day we forecast how much the Nifty will move over the next 15 minutes, the next 30 minutes, or the last hour of the day (14:30 to 15:30). "How much it moves" is the realized variance: the sum of squared 1-minute log returns in that window. Forecasts are made every 15 minutes from the open for the 15 and 30 minute targets (about 24 forecasts per day), and once per day at 14:30 for the last hour.

**The benchmark.** mcsGARCH is the model of the anchor paper (Chhimwal, Pandey and Pandey 2025). It multiplies three parts: a daily variance level (from an EGARCH model on daily returns), an average intraday shape (volatility is high at the open and close, low at lunch), and a short-memory intraday GARCH term. Every other model is compared with it.

**QLIKE.** The main loss. For a realized value y and a forecast f, QLIKE = y/f - log(y/f) - 1. It is zero for a perfect forecast and punishes forecasts that are too low more than forecasts that are too high. Patton (2011) shows it ranks variance forecasts correctly even though realized variance is a noisy measure. Tables report QLIKE **relative to mcsGARCH in percent**: -5% means the model's average loss is 5% lower than mcsGARCH's, so the model is better.

**MSE.** The second loss: the average of (forecast minus realized) squared. It is reported next to QLIKE because the two can disagree. MSE is dominated by the few windows with the largest variance (one wild morning can outweigh a year of quiet ones), while QLIKE looks at the ratio of realized to forecast, so every window counts in proportion. A model that is roughly right on most windows but badly too high on one crash morning can win on QLIKE and lose on MSE.

**Trimmed QLIKE.** QLIKE has a long tail too: a forecast far too low on a sudden move gives a single huge loss. As a robustness check, the scoreboards also give the mean QLIKE after removing each model's own largest 1% of losses.

**Diebold-Mariano (DM) test.** Is the difference in average loss between two models larger than chance? Losses are first averaged within each day (forecasts within one day are not independent), so the test uses one number per day. The variance uses a Newey-West (HAC) correction for serial correlation across days, with the Harvey, Leybourne and Newbold small-sample factor, and p-values come from a Student t distribution with n - 1 degrees of freedom (n = number of days), as they recommend. We report one-sided p-values: "p = 0.01" means a difference this large in favour of the second model would happen by chance about 1% of the time if the two models were equally good.

**Holm adjustment.** When many tests are run together (for example, four GEX sets times three targets), some will look significant by luck. Holm's method raises the p-values so that the chance of even one false alarm in a family of tests stays at 5%. The ablation files give both the raw p-value and the Holm-adjusted one.

**Model Confidence Set (MCS).** The set of models that cannot be statistically separated from the best one (Hansen, Lunde and Nason 2011), at the 10% level, from a block bootstrap of daily losses.

**Walk-forward.** The first 150 days are only used for training. Then every model, mcsGARCH included, is refitted on all days before a block of 21 trading days and forecasts that block, block after block, to the end. (The first real-data run refitted mcsGARCH only every 63 days; see Iteration 10.) No model ever sees data from after the forecast time. Option data published at the close of day t-1 (bhavcopy, participant file, VIX) enter only forecasts for day t.

**Development and hold-out.** To keep myself honest I split the out-of-sample period in two before any modelling. Every design decision in Section 3 was made by looking only at test days up to 30 June 2025 (the "development" window: 462 test days, 99 of them expiry days). For those runs the data were cut at 30 June 2025, so the later days were never even computed. The days from 1 July 2025 to 2 November 2026 (the "hold-out") were evaluated once, with the final configuration. If the hold-out numbers look like the development numbers, the design choices did not just fit noise.

**GEX ablation.** The test of whether option positioning helps: the same model, the same data and the same training, fitted once without and once with a block of gamma exposure (GEX) features. If the version with GEX has lower out-of-sample loss, and the DM test says the gap is not chance, positioning carries information the base model lacks.

## 3. The iterations

### Iteration 0. Starting point: the first real-data run

The first real run (`results/pipeline_nse/`, rerun after the strike-sorting fix described in Iteration 1) used a base feature set of past realized variance at several horizons, VIX, time of day and the expiry calendar. On the development window (file `results/pipeline_nse_v2/development/dev_scoreboards.csv`, run `baseline_v1`):

| Model (first run) | next 15 min, all | next 15 min, expiry | next 30 min, all | next 30 min, expiry | last hour, all | last hour, expiry |
| --- | --- | --- | --- | --- | --- | --- |
| HAR | +8.3% | +7.0% | +8.0% | +5.3% | -0.9% | -3.3% |
| VIX | +81.6% | +51.5% | +90.0% | +53.8% | +72.6% | +19.3% |
| LightGBM, no GEX | -0.6% | -3.7% | -0.2% | -5.2% | +4.9% | -16.6% |
| Hybrid, no GEX | -6.1% | -4.5% | -7.5% | -4.6% | -1.4% | +2.2% |
| Hybrid + all GEX features | -5.8% | -3.0% | -7.5% | -4.0% | -2.6% | +0.0% |

So the hybrid (mcsGARCH multiplied by a LightGBM correction) clearly beat the paper's model at 15 and 30 minutes (DM p below 0.001), GEX features moved it by at most 2.2 points either way, and nothing was significant for the last hour. The open question was whether the GEX null was fair: a base model without option-implied information might let GEX "work" only as a stand-in for implied volatility, or a weak base might hide a real effect behind noise.

### Iteration 1. Three data problems found on the way

1. **Theoretical settlement prices.** NSE gives every listed option a settlement price, even if it did not trade that day; for untraded series that price is a theoretical value (in the files we checked, a nearly flat 15 to 17% volatility). Far weekly expiries are often untraded. When I first computed a one-month implied volatility from all listed series, it jumped by 3 to 5 volatility points from day to day and its skew was exactly zero on many days. Fix: implied quantities are computed only from series that traded that day, and an expiry enters the term structure only if its volume is at least 0.5% of the nearest expiry's. After the fix, the one-month implied volatility tracks India VIX with a correlation of 0.93 on the development days (`dev_implied_checks.csv`).
2. **Unsorted UDiFF files** (found by the coordinating thread, applied here). Bhavcopies from July 2024 are not ordered by strike, so the old GEX code filled missing implied vols from the wrong neighbouring strikes. The loader now sorts by strike, and the GEX code sorts again before it interpolates.
3. **Holiday-shifted expiries** (found by the coordinating thread). Contracts listed for a holiday (29 June 2023) expired the day before. The option tables used for implied features now use the same corrected expiry dates as the chains.

### Iteration 2. Option-implied information in the base

**What.** A new module `src/niftygex/implied.py` reads every listed Nifty option in the bhavcopy published at the close of day t-1 and computes, for use on day t only:

* the ATM implied volatility of the nearest expiry (the average of the call and put implied vols at the forward, which is what an ATM straddle prices), from Black-76 with the put-call parity forward of that expiry;
* the straddle-implied expected variance until expiry, iv^2 x T, with T counted in trading days; and its per-day version iv^2 / 252. On the day before an expiry, the nearest expiry is one trading day away, so this per-day number is the market's own forecast of the expiry day's variance;
* the second expiry's implied vol, a constant-maturity 21-trading-day (about one month) implied vol interpolated in total variance across liquid expiries, the term-structure slope log(near vol / one-month vol), and the implied forward variance per day between the nearest and second expiries;
* a skew proxy (implied vol one implied standard deviation below the forward minus one standard deviation above, divided by ATM vol) and a smile curvature proxy, for the nearest and the one-month expiry;
* put/call ratios of open interest (nearest expiry and all expiries) and of volume (nearest expiry);
* the implied-minus-realized variance gap, and how far the index has moved since the open in implied daily standard deviations.

A calendar block was added too: weekday, monthly-expiry day, day before expiry, Bank Nifty expiry day (when it is not a Nifty expiry), an expiry x time-of-day interaction, and the expiry regime (several weeklies before 20 November 2024, one Thursday weekly to 31 August 2025, Tuesday after).

**Checks** (`results/pipeline_nse_v2/development/dev_implied_checks.csv`, development test days): the straddle-implied variance agrees with iv^2 x T to within 1% on average (ratio 1.008). The log of the per-day implied variance from yesterday's chain correlates 0.615 with the log of today's realized variance, against 0.483 for India VIX. On expiry days the gap is similar, 0.571 against 0.409. So the nearest-expiry straddle is a better one-day-ahead volatility signal than VIX, which is a 30-day measure.

**Timing test.** `tests/test_implied.py` changes everything published at the close of day k (settlement prices, open interest, volume) and checks that every feature row for days up to k is unchanged and that the rows for day k+1 do change. This is the "features at day t use only the chain published at day t-1" guarantee.

### Iteration 3. New models that use the implied information

**What.** Three additions to the forecast ladder:

* `implied`: the straddle-implied per-day variance spread over the day with the diurnal curve, with one scale factor for expiry days and one for other days, fitted on the training window. The simplest possible option-based forecast, built like the existing VIX model.
* `mcsgarch_iv`: the paper's mcsGARCH with its daily component replaced by the straddle-implied per-day variance. This needs no new parameter, because the diurnal curve is estimated as the average of squared returns divided by the daily component, so it absorbs any constant scale. It answers a simple question: does the paper's own model improve if its daily level comes from the option market instead of an EGARCH on past returns?
* `linear_*`: a ridge regression of log variance on exactly the same features as the tree models (a well-specified log-HAR), with the penalty chosen inside the training window. This gives an honest benchmark for LightGBM.

The tree models now come in three families with the same features: `lgbm` (LightGBM on log variance), `hybrid` (mcsGARCH x exp(LightGBM on the log residual)) and `hybridiv` (the same on top of `mcsgarch_iv`). "v1" means the first run's base features, "base" means the new base.

**Result on the development window** (QLIKE vs mcsGARCH; `dev_scoreboards.csv`, run `dev2`, single seed):

| Model | next 15, all | next 15, expiry | next 30, all | next 30, expiry |
| --- | --- | --- | --- | --- |
| implied | +56.6% | +39.3% | +58.9% | +37.1% |
| mcsgarch_iv | -3.5% (p 0.07) | -1.7% | -5.0% (p 0.03) | -3.9% (p 0.10) |
| linear, v1 base | +0.2% | -0.5% | -1.9% | -1.7% |
| linear, new base | -2.4% | -1.7% | -4.6% (p 0.03) | -3.4% |
| LightGBM, v1 base | -0.5% | -3.1% | +0.3% | -5.2% |
| LightGBM, new base | -2.2% | -3.4% | -0.8% | -4.7% |
| hybrid, v1 base | -6.1% | -4.0% | -7.8% | -4.7% |
| hybrid, new base | -6.0% | -4.4% | -7.8% | -5.9% |
| hybridiv, new base | -6.8% | -3.6% | -8.7% | -5.3% |

What this shows:

* The straddle alone (`implied`) is a poor intraday forecast, because it knows nothing about how the day is going. Like VIX, it needs the intraday dynamics.
* The paper's model gets 3 to 5% better just by taking its daily level from the straddle (`mcsgarch_iv`).
* The implied block helps the linear model a lot (2.6 to 2.8 points; DM p 0.005 and 0.007 in `dev_ablation_single_seed.csv`), but it does not help the tree hybrid on average: the hybrid already corrects the daily level from realized variance and VIX.

### Iteration 4. A cleaner GEX ablation

**What.** The first run had two GEX sets that shared many unsigned features, so it could not tell "gamma size" from "gamma sign". The new sets form a ladder:

* `gross`: only unsigned measures: gross gamma relative to its 20-day median, gross gamma of strikes within one implied standard deviation of the forward ("near the money", where hedging flows concentrate), gross gamma divided by the day's Nifty futures traded value (gamma per unit of market depth: the share of a day's futures turnover that a 1% move would make hedgers trade), the share of gamma within 1% of spot, distances to the largest open-interest strike, max pain and the call and put walls, and the change in open interest.
* `us`, `participant`, `client`: `gross` plus the signed net GEX of one sign rule (all strikes, near the money, and per unit of futures turnover) and, where defined, the distance to the gamma flip. US convention: dealers long calls and short puts. Participant: dealers' net position taken from the "Pro" group in NSE's participant file. Client: dealers take the opposite side of the "Client" group.
* `placebo`: the `us` set's columns taken from a randomly chosen other day (same time of day). It has the same distribution as real GEX but no information about the day being forecast. Any "gain" it shows is noise, and its cost tells us how much simply adding a dozen columns hurts.

Comparisons: base vs base+set (does the set help?), gross vs signed (does the sign add anything beyond size?), placebo vs real `us` (same columns, real day or not).

Two facts from the final panel (`results/pipeline_nse_v2/daily_gex_panel.csv`, all 942 days): on the median day, strikes within one implied standard deviation of the forward carry 71% of all gross gamma; and the US-convention and Pro-signed net GEX have the same sign on only 45% of days. The sign rules are not small variations of one measure: they often say opposite things. The GEX columns shared with the first run match it exactly, except on four days around the 29 June 2023 holiday, where the holiday fix changed the time to expiry.

**Result on the development window** (single seed; change in QLIKE from adding the set, one-sided DM p; `dev_ablation_single_seed.csv`):

| Family, target | gross | US sign | participant (Pro) | client | placebo |
| --- | --- | --- | --- | --- | --- |
| hybrid, next 30 | -0.14% (0.33) | +0.10% (0.62) | -0.97% (0.007) | +0.01% (0.52) | +0.58% (0.98) |
| hybrid, next 15 | +0.35% (0.93) | 0.00% (0.50) | -0.06% (0.39) | +0.22% (0.73) | +0.36% (0.92) |
| hybridiv, next 30 | +0.34% (0.87) | +0.75% (0.93) | +0.10% (0.63) | +0.06% (0.60) | +0.81% (0.98) |
| hybridiv, next 15 | +0.31% (0.86) | -0.38% (0.10) | +0.09% (0.62) | +0.06% (0.62) | -0.06% (0.34) |
| linear, next 30 | +0.37% (0.69) | -0.49% (0.34) | +1.31% (0.86) | +0.54% (0.77) | +0.65% (0.99) |

One cell looked like a discovery: Pro-signed GEX improving the hybrid by about 1% at 30 minutes (p 0.007). Iteration 7 shows it was not robust.

### Iteration 5. Regimes: the implied edge is concentrated before November 2024

Breaking the development window by expiry regime (`dev_scoreboards.csv`, run `dev2`, next 30 minutes, QLIKE vs mcsGARCH):

| Model | several weeklies (to 19 Nov 2024; 312 days) | one Thursday weekly (20 Nov 2024 to 30 Jun 2025; 150 days) |
| --- | --- | --- |
| mcsgarch_iv | -8.3% | +0.5% |
| linear, new base | -9.8% | +4.0% |
| hybridiv, new base | -10.4% | -5.9% |
| hybrid, new base | -7.8% | -7.8% |

Everything that leans on the straddle's level lost its edge after SEBI's November 2024 changes, while the hybrid on the paper's model held steady. One reason is visible in the data (`dev_implied_checks.csv`): on expiry days realized variance was 0.57 and 0.61 of the straddle-implied variance in the two earlier sub-periods but 0.47 after 20 November 2024, while non-expiry days stayed between 0.51 and 0.72. The expiry-day straddle became relatively more expensive, and a model with an expanding training window adapts only slowly to such a shift.

**Decision (made here, on development data only):** the headline GEX ablation uses the `hybrid` family: the paper's mcsGARCH times a LightGBM correction, with the full new base (which includes every implied feature and the `mcsgarch_iv` forecast as inputs). Reasons: it is the design the project planned from the start, it is stable across regimes, and its development gap to `hybridiv` (1.0 point at 30 minutes, p 0.24) is not significant. `hybridiv` and `linear` are reported as robustness families.

### Iteration 6. The last hour: train on every 60-minute window, test on 14:30

**Problem.** The last-hour target has one observation per day, so a training window has only 150 to 900 rows. In the first run and in development, the tree models were erratic there: on expiry days plain LightGBM was 24.6% better than mcsGARCH but 15.7% worse on other days, and the placebo GEX set moved the hybrid's score by 2.8% (by 8.7% on expiry days for `hybridiv`).

**Change.** Train the last-hour models on all 60-minute windows of each day (starting every 15 minutes, 22 per day) and forecast only the window that starts at 14:30. The test set is identical; the training set is 22 times larger, and time of day is a feature, so the model can still learn what is special about the last hour. Implemented as `eval_minutes` in `WalkForwardConfig`.

**Result on the development window** (`dev_scoreboards.csv`, targets `last_hour` and `last_hour_pooled`): the placebo's effect on the hybrid fell from +2.8% to -0.2% (and from +3.9% to +0.2% for `hybridiv`), so estimator noise dropped sharply. The linear model went from -1.7% to -7.1% against mcsGARCH. The hybrid was unchanged (-2.9% and -2.8%), and nothing beats mcsGARCH significantly in the last hour in either version. Kept, as a variance reduction with no cost.

### Iteration 7. Model noise: random seeds and tuning

**Seeds.** LightGBM uses random subsamples of rows and columns, so a different random seed gives a slightly different model. I reran several tree models with seeds 1 and 2 on the development window (next 30 minutes; `dev_seed_noise_next30.csv`):

* The same model's QLIKE moved by 0.3 to 0.55 points across seeds.
* A DM test between two seeds of the *identical* model gave p = 0.028 once (Pro-GEX hybrid, seed 0 vs seed 1). The DM test treats each model as fixed, so it does not know about this randomness.
* The Pro-GEX "discovery" of Iteration 4 (-0.97%, p 0.007 with seed 0) became -0.10% (p 0.36) with seed 1 and -0.42% (p 0.06) with seed 2.

**Decision:** every LightGBM forecast in the final run is the average (in logs) of 5 models with different seeds, so that a with/without-GEX comparison measures the features, not the luck of the draw. Reported in the final run as `seed_noise_next30.csv`, with single-seed reruns for calibration.

**Tuning inside the training window.** With 3-seed averages, I compared the default hyperparameters (15 leaves, at least 100 rows per leaf) with tuning on the last 20% of each training window over a 3 x 3 grid (`dev_tuning_next30.csv`). Tuning made plain LightGBM much better (-1.2% to -5.2% against mcsGARCH) but made both base hybrids slightly worse (+0.4 points for `hybrid`, +0.5 for `hybridiv`, not significant). It also flipped the GEX ablation: Pro sign -0.64% (p 0.03) with defaults but -0.10% (p 0.46) when tuned; US sign +0.13% (p 0.67) with defaults but -0.59% (p 0.03) when tuned. A positioning "effect" that changes sign with the hyperparameters is not an effect.

**Decision:** the hybrids keep the pre-specified defaults. A tuned plain LightGBM (`lgbmt_base`) is added so that the comparison with the ridge log-HAR is fair to LightGBM.

### Iteration 8. The closing auction (from 3 August 2026)

The coordinating thread found that from 3 August 2026 NSE sets the official close in a closing auction: in the 1-minute data the index is frozen from 15:16 to 15:28 and the 15:29 bar jumps to the auction price. This changes every window that covers the end of the day, including the whole last-hour target. Two changes: a `closing_auction` indicator in the calendar block (a public rule known in advance, not something learned from returns), and every scoreboard and ablation is also reported on test days before 3 August 2026 (groups `pre_auction`, `pre_auction_expiry`, `holdout_pre_auction`). This change came from a data check, not from looking at my hold-out results. All 63 auction days fall in the hold-out.

### Iteration 9. A ridge safeguard found on synthetic data

A short synthetic smoke test of the new pipeline (simulated market, not real data) showed one ridge model 168% worse than its base: on two test days some features fell far outside their training range, and a linear model extrapolates without limit. Fix: each test feature is clipped to the range it had in the training window. On the same synthetic run the gap went from +168% to -0.01%.

### Iteration 10. Changes after an independent review

After the development work, an outside reviewer read the first real-data results and the code (`docs/INDEPENDENT_REVIEW.md` on the main branch). Six of the points apply to this work. All six were handled before the final run, and none of them was decided by looking at hold-out results.

1. **Refit schedule (review M3).** The first run refitted mcsGARCH and its daily EGARCH every 63 trading days but the learned models every 21, which handicaps the paper's model a little. In the final run mcsGARCH (and `mcsgarch_iv`) is refitted every 21 days like everything else, and the old schedule is kept as an extra model, `mcsgarch_r63`, so you can see what the schedule does. Every "vs mcsGARCH" number in Section 4 is against the 21-day refit.
2. **MSE (review M1).** The scoreboards now give MSE relative to mcsGARCH next to QLIKE, with DM tests and a Model Confidence Set for both losses, and the GEX ablation for the hybrid is also given under MSE (`ablation_<target>_mse.csv`).
3. **Where the gain comes from (review M2 and m9).** Results are now broken down by anchor time (the 09:15 forecast, which has to absorb the overnight news, against all later anchors; `gain_by_anchor_<target>.csv` has every anchor), by expiry regime, and with a trimmed-mean QLIKE that removes each model's largest 1% of losses.
4. **A whole-sample constant (review m4).** The order-flow feature divides each 1-minute price change by a typical size of a 1-minute move, and that size was the median over the whole sample, so it used later data. It is a single scale number, so the effect is tiny, but it is now an expanding median over previous days only. While writing a test for this, I found the same kind of problem in my own placebo: the "other day" was drawn from all days, later ones included. It is now drawn from earlier days only. A new test (`test_feature_rows_do_not_change_when_later_days_are_removed` in `tests/test_forecast_v2.py`) rebuilds the whole feature table on a market with its last days deleted and checks that every numeric feature of every earlier day is unchanged.
5. **DM p-values (review m5).** Now from a Student t with n - 1 degrees of freedom instead of the normal. With 300 to 800 days this changes p-values only in the third decimal.
6. **The regime label as a feature (review m7).** The expiry regime code is a feature, so in the first 21-day block after each regime change the models have never seen its new value. Nothing leaks, but the feature cannot help exactly when it would matter: LightGBM treats the code as a number and sends the new value down the same branch as the previous regime, and the ridge model, whose inputs are clipped to their training range, treats the new regime like the first one. I did not replace it with the expiry weekday, as the reviewer suggests, because for Nifty that would not help: in the data, Nifty expired on Thursdays (Wednesday when Thursday was a holiday) until August 2025 and on Tuesdays (Monday after a holiday) from September 2025, so the expiry weekday changes value at exactly the same moment. The features that carry over across regimes are the ones that matter most anyway: expiry day or not, and trading days to expiry. The same holds for the `closing_auction` indicator: it is zero in every training window until the first refit after 3 August 2026.

One more change came from the coordinating thread: the monthly-expiry flag now comes from the expiry dates of Nifty index futures in the same bhavcopies, instead of "last expiry of the calendar month", which mislabelled a few weeklies moved by holidays.

The development tables in Iterations 0 to 9 were computed before these changes (63-day mcsGARCH refit, normal DM p-values, whole-sample order-flow scale, placebo from any day). They are left as they were, because they are the record of what I saw when I made each decision. Section 4 gives the development window again under the final protocol.

**What the changes did** (final run, Section 4). The 21-day refit makes mcsGARCH itself 1.6%, 1.8% and 2.1% better (the `mcsgarch_r63` row), so every challenger now faces a stronger benchmark. On the development window the hybrid's margin over mcsGARCH is 5.3% and 6.6% at 15 and 30 minutes under the final protocol, against 6.0% and 7.8% in Iteration 3 (that comparison also mixes in 5-seed averaging). In the last hour it went the other way: under the final protocol the hybrid beats mcsGARCH on the development window by 5.3% (p 0.048), where Iteration 6 found no significant gap. The new tables (MSE, 09:15 anchor, trimmed QLIKE, regimes) are in Section 4.4.

### What did not help during development, in one list

* The implied block did not improve the tree hybrid on average (it did improve the linear model and the paper's model).
* Hyperparameter tuning did not improve the hybrids.
* `hybridiv` (the hybrid on the implied-variance mcsGARCH) was best overall on development but lost its edge after November 2024 and was poor in the last hour on expiry days (+10 to +11% against mcsGARCH in the development runs).
* The `implied` model on its own is far worse than mcsGARCH intraday (+57 to +69% in the development runs).
* No GEX set gave a gain that survived a change of random seed or of hyperparameters on the development window.

The final run (Section 4) confirms the first, fourth and fifth points, and that `hybridiv` is no better than `hybrid` overall and poor in the last hour on expiry days (although in the Tuesday-expiry regime it did regain an edge over mcsGARCH). It did not repeat the tuning comparison for the hybrids.

## 4. Final results (REAL DATA)

The final configuration, fixed before the hold-out was computed: mcsGARCH and every learned model refitted every 21 trading days on an expanding window (first forecast after 150 training days); every LightGBM forecast averaged over 5 seeds; the hybrids with the pre-specified default hyperparameters, plain LightGBM also in a version tuned inside each training window; last-hour models trained on all 60-minute windows and evaluated at 14:30 only; DM tests with a Student t. The run took 99 minutes on one core (`run_info.json`).

Test days: 792 in all (169 expiry days); 729 before the closing auction (156 expiry days) and 63 in the auction period (13 expiry days); 462 in the development window (99 expiry days) and 330 in the hold-out (70 expiry days). Every cell is QLIKE relative to mcsGARCH (negative = better than mcsGARCH), with the one-sided DM p-value that the model has lower loss than mcsGARCH in brackets. Files: `scoreboard_<target>.csv` and `tests_<target>.csv`.

### 4.1 Scoreboard, all test days and expiry days (REAL DATA)

| Model | 15 min, all days | 15 min, expiry days | 30 min, all days | 30 min, expiry days | last hour, all days | last hour, expiry days |
| --- | --- | --- | --- | --- | --- | --- |
| mcsGARCH refitted every 63 days | +1.6% (0.999) | +1.0% (0.947) | +1.8% (1.000) | +1.2% (0.984) | +2.1% (0.996) | +2.3% (0.958) |
| HAR | +6.1% (0.999) | +4.6% (0.889) | +7.9% (1.000) | +5.7% (0.987) | +1.9% (0.717) | +10.2% (0.868) |
| VIX | +53.9% (1.000) | +34.9% (1.000) | +65.8% (1.000) | +44.1% (1.000) | +20.4% (0.930) | +32.2% (0.938) |
| implied (straddle) | +41.6% (1.000) | +30.1% (1.000) | +48.1% (1.000) | +35.1% (1.000) | +26.2% (0.993) | +60.9% (0.999) |
| mcsGARCH, implied daily level | -1.0% (0.250) | +1.4% (0.671) | -1.9% (0.142) | +0.2% (0.521) | -1.2% (0.345) | +5.1% (0.711) |
| ridge log-HAR, v1 base | -1.0% (0.304) | +0.4% (0.541) | -2.4% (0.071) | +0.3% (0.538) | -8.5% (0.004) | -11.2% (0.041) |
| ridge log-HAR, new base | -2.2% (0.104) | -1.0% (0.401) | -3.9% (0.013) | -1.7% (0.295) | -7.0% (0.011) | -6.4% (0.171) |
| LightGBM, v1 base | -3.0% (0.155) | -6.3% (0.003) | -2.6% (0.304) | -5.9% (<0.001) | -6.1% (0.080) | -6.9% (0.058) |
| LightGBM, new base | -3.6% (0.077) | -6.3% (0.003) | -3.5% (0.206) | -6.6% (0.001) | -6.2% (0.085) | -2.4% (0.357) |
| LightGBM tuned, new base | -5.1% (<0.001) | -6.2% (0.004) | -5.9% (0.003) | -6.7% (0.001) | -7.2% (0.040) | -3.1% (0.326) |
| hybrid, v1 base | -5.5% (<0.001) | -5.8% (0.001) | -7.8% (<0.001) | -5.5% (<0.001) | -6.0% (<0.001) | -7.7% (0.025) |
| **hybrid, new base** | -5.8% (<0.001) | -6.6% (0.001) | -7.6% (<0.001) | -6.2% (<0.001) | -6.4% (<0.001) | -7.9% (0.010) |
| hybridiv, new base | -5.6% (<0.001) | -4.2% (0.002) | -7.3% (<0.001) | -4.9% (0.005) | -2.7% (0.217) | +3.8% (0.622) |

The bold row is the headline model chosen in Iteration 5. Reading the table: the paper's model is hard to beat with simple tools (HAR, VIX, the straddle alone and the 63-day refit are all worse), the option-implied daily level is about as good as the paper's EGARCH level, and the clear winners at 15 and 30 minutes all use LightGBM. In the last hour the ridge log-HAR is as good as the hybrid (7.0% to 8.5% better than mcsGARCH, p 0.004 to 0.011 on all days). The hybrid is the only model with p at most 0.010 against mcsGARCH in all six columns.

### 4.2 Before the closing auction, and the auction period (REAL DATA)

The closing auction from 3 August 2026 is treated as its own regime (Iteration 8). On the 729 test days before it:

| Model | 15 min, all days | 15 min, expiry days | 30 min, all days | 30 min, expiry days | last hour, all days | last hour, expiry days |
| --- | --- | --- | --- | --- | --- | --- |
| mcsGARCH refitted every 63 days | +1.0% (0.999) | +0.5% (0.900) | +1.4% (0.999) | +0.9% (0.961) | +1.6% (0.961) | +2.4% (0.932) |
| HAR | +9.0% (1.000) | +9.0% (1.000) | +9.1% (1.000) | +7.0% (0.998) | +4.0% (0.792) | +12.6% (0.867) |
| VIX | +64.6% (1.000) | +43.3% (1.000) | +75.0% (1.000) | +48.9% (1.000) | +59.7% (1.000) | +39.8% (0.931) |
| implied (straddle) | +49.4% (1.000) | +35.8% (1.000) | +55.3% (1.000) | +38.4% (1.000) | +60.7% (1.000) | +62.6% (0.998) |
| mcsGARCH, implied daily level | -0.7% (0.341) | +2.4% (0.745) | -1.8% (0.184) | +0.4% (0.552) | +0.2% (0.519) | +7.5% (0.740) |
| ridge log-HAR, v1 base | +1.0% (0.704) | +3.8% (0.823) | -1.0% (0.269) | +1.6% (0.696) | -6.8% (0.029) | -9.2% (0.094) |
| ridge log-HAR, new base | -0.9% (0.309) | +2.2% (0.726) | -3.0% (0.049) | -0.6% (0.425) | -7.6% (0.035) | -4.7% (0.278) |
| LightGBM, v1 base | -2.0% (0.273) | -5.4% (0.005) | -0.8% (0.441) | -5.0% (0.003) | -1.9% (0.355) | -6.8% (0.108) |
| LightGBM, new base | -2.3% (0.197) | -4.5% (<0.001) | -1.7% (0.356) | -5.5% (0.005) | -1.2% (0.410) | -0.6% (0.472) |
| LightGBM tuned, new base | -3.9% (<0.001) | -4.5% (0.002) | -4.5% (0.023) | -5.5% (0.006) | -3.7% (0.209) | +0.4% (0.521) |
| hybrid, v1 base | -5.5% (<0.001) | -5.4% (<0.001) | -6.7% (<0.001) | -4.7% (0.004) | -5.8% (0.008) | -7.0% (0.067) |
| **hybrid, new base** | -5.5% (<0.001) | -5.4% (<0.001) | -6.9% (<0.001) | -5.4% (0.001) | -7.0% (0.002) | -6.5% (0.036) |
| hybridiv, new base | -5.2% (<0.001) | -3.2% (0.005) | -6.7% (<0.001) | -4.4% (0.016) | -1.6% (0.382) | +7.4% (0.682) |

The picture is the same as on all days. The 63 auction days (all in the hold-out) are a different world: the frozen index from 15:16 to 15:28 and the jump at 15:29 change the end of the day. Almost every model beats mcsGARCH there, which fits the fact that mcsGARCH's time-of-day shape is an average over all earlier days, nearly all of them from before the auction. The hold-out split into the days before the auction and the auction period:

| Model | 15 min, hold-out before auction | 15 min, auction period | 30 min, hold-out before auction | 30 min, auction period | last hour, hold-out before auction | last hour, auction period |
| --- | --- | --- | --- | --- | --- | --- |
| mcsGARCH refitted every 63 days | +0.2% (0.852) | +4.9% (0.991) | +0.1% (0.799) | +4.5% (0.997) | -0.3% (0.136) | +3.1% (0.990) |
| HAR | +8.5% (1.000) | -9.8% (0.090) | +8.7% (1.000) | -1.1% (0.397) | +1.6% (0.598) | -1.6% (0.328) |
| mcsGARCH, implied daily level | +2.4% (0.803) | -2.7% (<0.001) | +1.3% (0.701) | -2.6% (0.004) | -7.5% (0.068) | -3.7% (0.022) |
| ridge log-HAR, new base | -0.3% (0.449) | -9.7% (0.035) | -3.2% (0.043) | -10.7% (0.031) | -13.6% (0.003) | -5.8% (0.054) |
| LightGBM, new base | -5.9% (<0.001) | -11.1% (0.055) | -7.8% (<0.001) | -16.4% (0.010) | -9.7% (0.002) | -14.8% (0.034) |
| LightGBM tuned, new base | -6.3% (<0.001) | -11.5% (0.051) | -8.0% (<0.001) | -16.3% (0.008) | -11.9% (0.010) | -13.1% (0.039) |
| **hybrid, new base** | -5.9% (<0.001) | -7.4% (0.088) | -7.4% (<0.001) | -13.5% (0.003) | -10.1% (0.002) | -5.2% (0.004) |
| hybridiv, new base | -4.6% (<0.001) | -7.4% (0.060) | -6.7% (<0.001) | -11.5% (0.007) | -9.7% (0.010) | -4.6% (0.019) |

With 63 days (13 expiry days) the auction-period numbers are noisy; they are reported, not interpreted.

### 4.3 Development window and hold-out (REAL DATA)

| Model | 15 min, development | 15 min, hold-out | 30 min, development | 30 min, hold-out | last hour, development | last hour, hold-out |
| --- | --- | --- | --- | --- | --- | --- |
| mcsGARCH refitted every 63 days | +1.4% (1.000) | +1.8% (0.976) | +2.0% (0.999) | +1.4% (0.985) | +2.6% (0.972) | +1.8% (0.970) |
| HAR | +9.2% (1.000) | +2.3% (0.785) | +9.3% (0.998) | +5.8% (0.999) | +5.3% (0.780) | -0.3% (0.459) |
| mcsGARCH, implied daily level | -2.3% (0.156) | +0.6% (0.629) | -3.2% (0.110) | +0.2% (0.534) | +4.6% (0.748) | -5.2% (0.011) |
| ridge log-HAR, new base | -1.2% (0.304) | -3.5% (0.087) | -2.9% (0.116) | -5.5% (0.009) | -4.2% (0.241) | -8.8% (0.001) |
| LightGBM, new base | -0.4% (0.459) | -7.7% (0.001) | +1.1% (0.562) | -10.4% (<0.001) | +3.7% (0.670) | -12.9% (0.007) |
| LightGBM tuned, new base | -2.7% (0.041) | -8.1% (<0.001) | -3.0% (0.184) | -10.5% (<0.001) | +0.9% (0.552) | -12.6% (0.006) |
| **hybrid, new base** | -5.3% (<0.001) | -6.4% (<0.001) | -6.6% (<0.001) | -9.2% (<0.001) | -5.3% (0.048) | -7.1% (<0.001) |
| hybridiv, new base | -5.6% (<0.001) | -5.6% (<0.001) | -6.7% (<0.001) | -8.1% (<0.001) | +3.1% (0.654) | -6.6% (0.001) |

The hybrid's margin is a little larger in the hold-out than in the development window, so the design choices made on development data did not just fit noise. Plain LightGBM, by contrast, was no better than mcsGARCH on development days and much better in the hold-out: its edge depends on the period (Section 4.4, regimes).

### 4.4 Where the gain comes from (review items M1, M2, m9; REAL DATA)

**By anchor time.** The 09:15 forecast has to absorb the overnight news through the opening print, and mcsGARCH handles it with a fixed time-of-day factor times yesterday's daily level. Gains there are large; at all later anchors they are smaller but still clear for the hybrid:

| Model | 15 min, 09:15 only | 15 min, without 09:15 | 30 min, 09:15 only | 30 min, without 09:15 |
| --- | --- | --- | --- | --- |
| HAR | +17.7% (0.995) | +5.0% (0.996) | +8.7% (0.915) | +7.8% (1.000) |
| mcsGARCH, implied daily level | +6.9% (0.672) | -1.8% (0.007) | +1.2% (0.531) | -2.2% (0.024) |
| ridge log-HAR, new base | +11.4% (0.785) | -3.5% (0.001) | -0.1% (0.498) | -4.3% (<0.001) |
| LightGBM, new base | -15.1% (<0.001) | -2.5% (0.183) | -22.2% (<0.001) | -1.5% (0.377) |
| LightGBM tuned, new base | -19.7% (0.005) | -3.7% (0.021) | -25.7% (<0.001) | -3.9% (0.096) |
| **hybrid, new base** | -13.3% (<0.001) | -5.1% (<0.001) | -23.9% (<0.001) | -6.0% (<0.001) |
| hybridiv, new base | -14.8% (0.056) | -4.7% (<0.001) | -24.9% (0.010) | -5.5% (<0.001) |

Selected anchor times (`gain_by_anchor_<target>.csv` has every anchor; the last anchor, 15:00, includes the close):

| Target, model | 09:15 | 09:30 | 09:45 | 10:15 | 11:15 | 12:15 | 13:15 | 14:15 | 15:00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15 min, HAR | +17.7% | +29.4% | +22.1% | +36.5% | +5.6% | +7.9% | -1.1% | +3.8% | -6.3% |
| 15 min, ridge log-HAR, new base | +11.4% | -9.4% | -3.9% | +3.7% | -0.7% | +2.8% | -5.6% | +0.0% | -13.0% |
| 15 min, LightGBM, new base | -15.1% | -10.6% | -4.8% | +0.5% | +1.1% | +1.5% | -4.6% | +1.5% | -11.5% |
| 15 min, hybrid, new base | -13.3% | -14.1% | -6.5% | -1.6% | -4.0% | -2.0% | -5.2% | -1.0% | -12.5% |
| 30 min, HAR | +8.7% | +13.1% | +55.7% | +26.5% | +9.3% | +14.8% | +4.2% | +2.8% | -10.7% |
| 30 min, ridge log-HAR, new base | -0.1% | -14.7% | -5.7% | +1.0% | -2.1% | +3.0% | -6.7% | -2.4% | -16.5% |
| 30 min, LightGBM, new base | -22.2% | -14.8% | -10.2% | -0.6% | +3.8% | +9.5% | -2.6% | +2.5% | -17.2% |
| 30 min, hybrid, new base | -23.9% | -17.2% | -10.4% | -3.5% | -4.2% | -2.3% | -6.0% | -2.6% | -14.8% |

The 09:15 anchor is 4% of all forecasts but brings 20% (15 minutes) and 29% (30 minutes) of the hybrid's total reduction in QLIKE; the other 80% and 71% come from the rest of the day. Plain LightGBM depends much more on the open (37% and 60% of its reduction), and without the 09:15 anchor its gain is small and not significant (2.5% and 1.5%). The ridge log-HAR is the opposite: no better than mcsGARCH at the open, but 3.5% and 4.3% better at the later anchors (p 0.001 and below 0.001).

**By expiry regime** (several weeklies to 19 November 2024: 312 test days; one Thursday weekly to 31 August 2025: 192; Tuesday weekly from 1 September 2025: 288):

| Model | 15 min, several weeklies | 15 min, Thursday weekly | 15 min, Tuesday weekly | 30 min, several weeklies | 30 min, Thursday weekly | 30 min, Tuesday weekly | last hour, several weeklies | last hour, Thursday weekly | last hour, Tuesday weekly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HAR | +9.0% (0.996) | +7.5% (0.995) | +2.5% (0.776) | +9.4% (0.976) | +7.5% (0.991) | +6.5% (0.999) | +4.1% (0.660) | +3.7% (0.688) | +0.3% (0.537) |
| mcsGARCH, implied daily level | -5.0% (0.013) | +2.3% (0.738) | +0.1% (0.526) | -6.1% (0.010) | +2.0% (0.683) | -0.5% (0.404) | -7.4% (0.177) | +17.6% (0.969) | -5.4% (0.012) |
| ridge log-HAR, new base | -5.9% (0.003) | +3.8% (0.849) | -3.2% (0.144) | -7.8% (<0.001) | +2.6% (0.760) | -5.1% (0.028) | -12.2% (0.077) | +4.0% (0.758) | -8.6% (0.003) |
| LightGBM, new base | +1.4% (0.590) | -3.9% (0.022) | -8.1% (0.003) | +4.8% (0.673) | -5.4% (0.009) | -11.1% (<0.001) | +0.8% (0.525) | +3.0% (0.667) | -12.7% (0.012) |
| LightGBM tuned, new base | -2.9% (0.064) | -3.2% (0.063) | -8.5% (0.002) | -1.6% (0.370) | -5.4% (0.010) | -11.1% (<0.001) | -4.4% (0.327) | +4.1% (0.727) | -12.5% (0.010) |
| **hybrid, new base** | -5.2% (<0.001) | -5.2% (<0.001) | -6.8% (0.002) | -6.7% (<0.001) | -6.4% (<0.001) | -9.7% (<0.001) | -9.9% (0.021) | +0.0% (0.507) | -7.1% (<0.001) |
| hybridiv, new base | -7.5% (<0.001) | -2.9% (0.096) | -5.7% (0.002) | -8.1% (0.001) | -4.6% (0.025) | -8.6% (<0.001) | -9.5% (0.124) | +16.3% (0.922) | -6.6% (0.002) |

The hybrid beats mcsGARCH in every regime at 15 and 30 minutes. Plain LightGBM does not (no better than mcsGARCH in the first regime), and the option-implied models (mcsGARCH with the implied level, ridge) help most in the first regime, not at all in the Thursday regime (Iteration 5 gives one reason), and partly again under Tuesday expiries (ridge at 30 minutes and in the last hour, the implied-level mcsGARCH in the last hour). In the last hour the hybrid gains nothing in the Thursday regime.

**Trimmed QLIKE.** The 1% largest anchor losses carry 26% to 34% of each model's total QLIKE (the largest single loss is 188 at 15 minutes, 115 at 30 minutes and 35 for the last hour, for mcsGARCH). Removing each model's own top 1% of losses:

| Model | 15 min, all days | 15 min, before auction | 30 min, all days | 30 min, before auction | last hour, all days | last hour, before auction |
| --- | --- | --- | --- | --- | --- | --- |
| mcsGARCH refitted every 63 days | +1.0% | +0.6% | +1.5% | +1.0% | +2.2% | +1.7% |
| HAR | +7.7% | +7.9% | +8.4% | +8.8% | +5.2% | +8.0% |
| mcsGARCH, implied daily level | -1.2% | -1.2% | -1.1% | -1.0% | +0.6% | +3.4% |
| ridge log-HAR, new base | -1.3% | -0.9% | -2.4% | -1.7% | -5.4% | -2.1% |
| LightGBM, new base | -4.5% | -4.2% | -5.5% | -4.7% | -4.0% | -1.7% |
| LightGBM tuned, new base | -4.6% | -4.2% | -5.7% | -4.9% | -4.2% | -2.1% |
| **hybrid, new base** | -5.6% | -5.4% | -6.7% | -6.1% | -6.4% | -4.9% |
| hybridiv, new base | -5.2% | -5.1% | -6.0% | -5.7% | -3.8% | +0.3% |

The ranking does not depend on the tail.

**MSE.** Relative to mcsGARCH (DM p-values on daily average squared errors):

| Model | 15 min, all days | 15 min, before auction | 30 min, all days | 30 min, before auction | last hour, all days | last hour, before auction |
| --- | --- | --- | --- | --- | --- | --- |
| mcsGARCH refitted every 63 days | +0.2% (0.945) | -0.0% (0.449) | +0.2% (0.960) | +0.0% (0.623) | +0.9% (0.993) | +1.1% (0.990) |
| HAR | -85.0% (0.149) | -84.7% (0.158) | -86.0% (0.150) | -85.8% (0.159) | +17.2% (0.639) | +24.9% (0.646) |
| VIX | -84.5% (0.149) | -84.2% (0.159) | -85.0% (0.151) | -84.7% (0.160) | +53.6% (0.795) | +76.4% (0.804) |
| mcsGARCH, implied daily level | -75.9% (0.180) | -77.5% (0.183) | -77.7% (0.182) | -79.4% (0.184) | +3.6% (0.776) | +5.0% (0.779) |
| ridge log-HAR, new base | -86.0% (0.149) | -86.0% (0.157) | -87.9% (0.150) | -87.8% (0.157) | -22.9% (0.127) | -30.4% (0.136) |
| LightGBM, new base | -84.9% (0.148) | -84.6% (0.158) | -85.8% (0.149) | -85.5% (0.158) | +26.9% (0.673) | +37.9% (0.677) |
| LightGBM tuned, new base | -85.3% (0.148) | -85.0% (0.157) | -86.2% (0.149) | -85.9% (0.158) | +21.8% (0.654) | +31.0% (0.659) |
| **hybrid, new base** | -33.2% (0.134) | -31.4% (0.155) | -26.9% (0.134) | -25.4% (0.154) | -18.3% (0.053) | -24.2% (0.060) |
| hybridiv, new base | -81.0% (0.164) | -81.4% (0.171) | -83.6% (0.164) | -84.3% (0.170) | -18.1% (0.052) | -24.0% (0.057) |

Under MSE the paper's model is the worst serious model at 15 and 30 minutes: HAR, VIX and plain LightGBM all have about 85% lower MSE. The reason is a handful of opening forecasts after big moves. At 15 minutes the ten largest squared errors are 97% of mcsGARCH's total; the largest is 09:15 on 5 June 2024, the morning after the election-result crash, where mcsGARCH forecast a variance of 0.00201 for a realized 0.000099 (twenty times too high). The hybrid forecast 0.00163 (it multiplies mcsGARCH by a correction, so it inherits the error), plain LightGBM 0.000008, HAR 0.000087, and the version of mcsGARCH whose daily level comes from the straddle 0.000228. That last comparison points at the daily component: the EGARCH daily level jumped after the crash, and the large 09:15 time-of-day factor multiplied it. QLIKE treats a forecast that is too high much more mildly than one that is too low, so the same windows barely matter for QLIKE. Two caveats make MSE a weak ranking here: none of the MSE differences against mcsGARCH at 15 and 30 minutes is significant (p 0.13 to 0.18, because they rest on a few days), and the MSE Model Confidence Set keeps 30 to 32 of the 32 models in every case (Section 4.6). Without the 09:15 anchor the ranking under MSE is the same in direction but much less extreme:

| Model | 15 min, without 09:15 | 30 min, without 09:15 |
| --- | --- | --- |
| HAR | -38.3% (0.189) | -35.1% (0.199) |
| mcsGARCH, implied daily level | -54.1% (0.177) | -57.3% (0.177) |
| ridge log-HAR, new base | -55.1% (0.154) | -54.1% (0.155) |
| LightGBM, new base | -37.2% (0.182) | -32.4% (0.201) |
| **hybrid, new base** | -25.8% (0.135) | -23.6% (0.129) |
| hybridiv, new base | -60.1% (0.162) | -64.0% (0.158) |

In the last hour, where there is no opening jump, MSE agrees more with QLIKE: the hybrid and ridge are about 18% to 23% better than mcsGARCH (p 0.05 to 0.13), and plain LightGBM is worse.

**In plain terms (review m10).** The hybrid's daily average loss is below mcsGARCH's on 71% of test days at 15 minutes, 74% at 30 minutes and 54% for the last hour. At 15 and 30 minutes this is a steady edge on most days, not a few lucky days; in the last hour it is a smaller edge built on fewer, larger wins.

**The refit schedule (review M3).** `mcsGARCH refitted every 63 days` is 1.6%, 1.8% and 2.1% worse than the 21-day refit on all days (rows above). The hybrids are built on the 21-day mcsGARCH, so their margins are against the stronger version.

### 4.5 The GEX ablation (REAL DATA)

The question of this project: does dealer gamma positioning add forecasting information to a strong model? Each cell is the change in QLIKE from adding a GEX set to the headline hybrid, with everything else identical, and the one-sided DM p-value (and the Holm-adjusted p-value within the four GEX sets of that row) that the GEX version is better. Negative means GEX helped. The placebo uses GEX of a random earlier day and has no Holm value (it is a control, not a hypothesis). Files: `ablation_<target>.csv`, figures `figures/ablation_<target>_<family>.png`.

**Hybrid, base versus base plus each GEX set:**

| Target, days | gross (unsigned) | US sign | Pro sign | client mirror | placebo |
| --- | --- | --- | --- | --- | --- |
| 15 min, all days | +0.14% (p 0.708; Holm 1.00) | -0.18% (p 0.266; Holm 1.00) | -0.10% (p 0.370; Holm 1.00) | +0.07% (p 0.637; Holm 1.00) | +0.20% (p 0.876) |
| 15 min, expiry days | +0.00% (p 0.505; Holm 1.00) | +0.43% (p 0.876; Holm 1.00) | -0.20% (p 0.325; Holm 1.00) | +0.20% (p 0.702; Holm 1.00) | +0.75% (p 0.995) |
| 15 min, before auction | -0.08% (p 0.279; Holm 0.56) | -0.28% (p 0.056; Holm 0.17) | -0.37% (p 0.026; Holm 0.11) | -0.07% (p 0.331; Holm 0.56) | +0.33% (p 0.995) |
| 15 min, before auction, expiry | +0.02% (p 0.519; Holm 1.00) | +0.55% (p 0.909; Holm 1.00) | -0.40% (p 0.181; Holm 0.72) | +0.21% (p 0.684; Holm 1.00) | +0.58% (p 0.977) |
| 15 min, development | +0.11% (p 0.750; Holm 1.00) | -0.07% (p 0.367; Holm 1.00) | -0.29% (p 0.117; Holm 0.47) | +0.14% (p 0.752; Holm 1.00) | +0.41% (p 0.993) |
| 15 min, hold-out | +0.19% (p 0.633; Holm 1.00) | -0.32% (p 0.297; Holm 1.00) | +0.15% (p 0.595; Holm 1.00) | -0.01% (p 0.485; Holm 1.00) | -0.07% (p 0.414) |
| 30 min, all days | -0.42% (p 0.024; Holm 0.09) | -0.33% (p 0.101; Holm 0.20) | -0.41% (p 0.045; Holm 0.14) | -0.32% (p 0.137; Holm 0.20) | +0.08% (p 0.612) |
| 30 min, expiry days | -0.36% (p 0.191; Holm 0.57) | +0.40% (p 0.801; Holm 0.80) | -0.84% (p 0.072; Holm 0.29) | -0.21% (p 0.321; Holm 0.64) | -0.17% (p 0.297) |
| 30 min, before auction | -0.24% (p 0.123; Holm 0.49) | -0.09% (p 0.362; Holm 0.72) | -0.27% (p 0.147; Holm 0.49) | -0.04% (p 0.441; Holm 0.72) | +0.34% (p 0.917) |
| 30 min, before auction, expiry | -0.17% (p 0.346; Holm 1.00) | +0.69% (p 0.930; Holm 1.00) | -0.72% (p 0.123; Holm 0.49) | +0.09% (p 0.583; Holm 1.00) | -0.02% (p 0.469) |
| 30 min, development | -0.06% (p 0.410; Holm 1.00) | +0.16% (p 0.703; Holm 1.00) | -0.27% (p 0.212; Holm 0.85) | -0.13% (p 0.359; Holm 1.00) | +0.45% (p 0.915) |
| 30 min, hold-out | -0.98% (p 0.002; Holm 0.01) | -1.10% (p 0.009; Holm 0.03) | -0.63% (p 0.024; Holm 0.05) | -0.62% (p 0.093; Holm 0.09) | -0.52% (p 0.115) |
| last hour, all days | +1.14% (p 0.815; Holm 1.00) | +0.92% (p 0.690; Holm 1.00) | +1.09% (p 0.783; Holm 1.00) | +1.59% (p 0.894; Holm 1.00) | +0.60% (p 0.844) |
| last hour, expiry days | +0.43% (p 0.610; Holm 1.00) | +1.63% (p 0.845; Holm 1.00) | +0.20% (p 0.542; Holm 1.00) | +0.97% (p 0.698; Holm 1.00) | +0.67% (p 0.685) |
| last hour, before auction | +0.06% (p 0.541; Holm 1.00) | +0.24% (p 0.638; Holm 1.00) | +0.21% (p 0.621; Holm 1.00) | +0.77% (p 0.794; Holm 1.00) | -0.11% (p 0.402) |
| last hour, before auction, expiry | -0.99% (p 0.190; Holm 0.76) | +0.43% (p 0.621; Holm 0.95) | -1.28% (p 0.225; Holm 0.76) | -0.12% (p 0.474; Holm 0.95) | -0.53% (p 0.344) |
| last hour, development | -0.49% (p 0.217; Holm 0.87) | -0.15% (p 0.425; Holm 0.90) | -0.44% (p 0.301; Holm 0.90) | +0.04% (p 0.513; Holm 0.90) | -0.43% (p 0.223) |
| last hour, hold-out | +2.26% (p 0.862; Holm 1.00) | +1.66% (p 0.705; Holm 1.00) | +2.15% (p 0.829; Holm 1.00) | +2.66% (p 0.914; Holm 1.00) | +1.30% (p 0.926) |

**Does the sign add anything beyond the size?** Signed set against the unsigned (gross) set, hybrid:

| Target, days | US sign | Pro sign | client mirror |
| --- | --- | --- | --- |
| 15 min, all days | -0.32% (p 0.054; Holm 0.11) | -0.24% (p 0.035; Holm 0.10) | -0.07% (p 0.289; Holm 0.29) |
| 15 min, expiry days | +0.42% (p 0.853; Holm 1.00) | -0.20% (p 0.232; Holm 0.70) | +0.20% (p 0.890; Holm 1.00) |
| 30 min, all days | +0.08% (p 0.680; Holm 1.00) | +0.01% (p 0.521; Holm 1.00) | +0.09% (p 0.672; Holm 1.00) |
| 30 min, expiry days | +0.76% (p 0.979; Holm 1.00) | -0.48% (p 0.142; Holm 0.43) | +0.15% (p 0.654; Holm 1.00) |
| last hour, all days | -0.22% (p 0.403; Holm 1.00) | -0.05% (p 0.459; Holm 1.00) | +0.45% (p 0.857; Holm 1.00) |
| last hour, expiry days | +1.19% (p 0.949; Holm 1.00) | -0.23% (p 0.400; Holm 1.00) | +0.53% (p 0.733; Holm 1.00) |

**Real against placebo** (same columns, real day or a random earlier day), hybrid:

| Target, days | US sign |
| --- | --- |
| 15 min, all days | -0.38% (p 0.168; Holm 0.17) |
| 15 min, expiry days | -0.32% (p 0.199; Holm 0.20) |
| 30 min, all days | -0.41% (p 0.064; Holm 0.06) |
| 30 min, expiry days | +0.58% (p 0.903; Holm 0.90) |
| last hour, all days | +0.33% (p 0.589; Holm 0.59) |
| last hour, expiry days | +0.95% (p 0.759; Holm 0.76) |

What this says, sign rule by sign rule, on all test days (15 min, 30 min, last hour; one-sided DM p in brackets):

* **US convention** (dealers long calls, short puts): -0.18% (0.266), -0.33% (0.101), +0.92% (0.690).
* **Pro participant** (dealer position from NSE's Pro category): -0.10% (0.370), -0.41% (0.045), +1.09% (0.783).
* **Client mirror** (dealers on the other side of clients): +0.07% (0.637), -0.32% (0.137), +1.59% (0.894).
* **Gross gamma, no sign**: +0.14% (0.708), -0.42% (0.024), +1.14% (0.815).

No cell passes Holm on all days, on expiry days or before the auction, and the signed sets do no better than unsigned gross gamma. The effects are a few tenths of a percent, the same size as the noise between two random seeds of the same model (Section 4.7).

**The one place where GEX looks better: the hold-out.** At 30 minutes in the hold-out, three sets pass Holm (gross -0.98%, US -1.10%, Pro -0.63%), and at 15 minutes in the hold-out before the auction all four sets are 0.4% to 0.7% better with raw p 0.015 to 0.051 (none passes Holm). I do not read this as evidence for positioning, for three reasons. First, the development window shows nothing (-0.3% to +0.2% at 30 minutes). Second, a large part of the 30-minute hold-out gain comes from the auction period, where the placebo "helps" just as much:

| Target, days | gross (unsigned) | US sign | Pro sign | client mirror | placebo |
| --- | --- | --- | --- | --- | --- |
| 15 min, hold-out before auction | -0.44% (p 0.037; Holm 0.11) | -0.69% (p 0.015; Holm 0.06) | -0.52% (p 0.051; Holm 0.11) | -0.48% (p 0.043; Holm 0.11) | +0.17% (p 0.819) |
| 15 min, auction period | +1.41% (p 0.816; Holm 1.00) | +0.42% (p 0.603; Holm 1.00) | +1.46% (p 0.807; Holm 1.00) | +0.89% (p 0.818; Holm 1.00) | -0.55% (p 0.285) |
| 30 min, hold-out before auction | -0.63% (p 0.024; Holm 0.10) | -0.63% (p 0.091; Holm 0.27) | -0.27% (p 0.220; Holm 0.44) | +0.16% (p 0.682; Holm 0.68) | +0.09% (p 0.637) |
| 30 min, auction period | -1.84% (p 0.006; Holm 0.02) | -2.27% (p 0.008; Holm 0.02) | -1.51% (p 0.002; Holm 0.01) | -2.56% (p 0.017; Holm 0.02) | -2.02% (p 0.052) |
| last hour, hold-out before auction | +1.07% (p 0.827; Holm 1.00) | +0.96% (p 0.772; Holm 1.00) | +1.43% (p 0.906; Holm 1.00) | +2.11% (p 0.947; Holm 1.00) | +0.48% (p 0.749) |
| last hour, auction period | +2.96% (p 0.824; Holm 1.00) | +2.07% (p 0.669; Holm 1.00) | +2.58% (p 0.770; Holm 1.00) | +2.98% (p 0.844; Holm 1.00) | +1.78% (p 0.912) |

Third, the sign never matters: unsigned gross gamma does as well as any signed version, although the US and Pro rules give the same sign on only 45% of days (Iteration 4). A real dealer-hedging effect should depend on the sign. What is left is a hint that the unsigned gamma features (how much option gamma there is, and how close to the money) carried a little information at 15 minutes in 2025 and 2026; it would need a new sample to test.

**Robustness families.** The same ablation for `hybridiv` (hybrid on the implied-level mcsGARCH) and for the ridge log-HAR, all days and expiry days:

`hybridiv`:

| Target, days | gross (unsigned) | US sign | Pro sign | client mirror | placebo |
| --- | --- | --- | --- | --- | --- |
| 15 min, all days | +0.12% (p 0.762; Holm 1.00) | -0.24% (p 0.106; Holm 0.42) | +0.18% (p 0.745; Holm 1.00) | -0.08% (p 0.334; Holm 1.00) | +0.36% (p 0.988) |
| 15 min, expiry days | -0.43% (p 0.194; Holm 0.52) | -0.75% (p 0.174; Holm 0.52) | -0.61% (p 0.118; Holm 0.47) | -0.46% (p 0.211; Holm 0.52) | +0.91% (p 0.957) |
| 30 min, all days | +0.05% (p 0.587; Holm 1.00) | +0.35% (p 0.788; Holm 1.00) | +0.42% (p 0.858; Holm 1.00) | -0.16% (p 0.259; Holm 1.00) | +0.24% (p 0.888) |
| 30 min, expiry days | -0.49% (p 0.100; Holm 0.40) | +0.38% (p 0.708; Holm 0.71) | -0.48% (p 0.134; Holm 0.40) | -0.48% (p 0.128; Holm 0.40) | +0.36% (p 0.723) |
| last hour, all days | -0.32% (p 0.409; Holm 1.00) | -0.61% (p 0.372; Holm 1.00) | +0.17% (p 0.553; Holm 1.00) | +0.04% (p 0.513; Holm 1.00) | +0.28% (p 0.670) |
| last hour, expiry days | -3.17% (p 0.190; Holm 0.57) | -2.90% (p 0.133; Holm 0.53) | -2.81% (p 0.230; Holm 0.57) | -1.76% (p 0.279; Holm 0.57) | +0.59% (p 0.889) |

Ridge log-HAR:

| Target, days | gross (unsigned) | US sign | Pro sign | client mirror | placebo |
| --- | --- | --- | --- | --- | --- |
| 15 min, all days | -0.13% (p 0.377; Holm 0.75) | -0.49% (p 0.247; Holm 0.74) | -0.03% (p 0.475; Holm 0.75) | -0.37% (p 0.172; Holm 0.69) | +0.12% (p 0.891) |
| 15 min, expiry days | -0.02% (p 0.490; Holm 1.00) | +1.55% (p 0.878; Holm 1.00) | +0.16% (p 0.619; Holm 1.00) | -0.08% (p 0.456; Holm 1.00) | -0.19% (p 0.122) |
| 30 min, all days | -0.33% (p 0.243; Holm 0.90) | -0.59% (p 0.224; Holm 0.90) | +0.97% (p 0.832; Holm 0.90) | -0.33% (p 0.253; Holm 0.90) | +0.30% (p 0.982) |
| 30 min, expiry days | -0.31% (p 0.327; Holm 1.00) | +1.04% (p 0.815; Holm 1.00) | -0.01% (p 0.495; Holm 1.00) | -0.13% (p 0.439; Holm 1.00) | -0.39% (p 0.057) |
| last hour, all days | -0.52% (p 0.241; Holm 0.48) | -0.78% (p 0.350; Holm 0.48) | -0.69% (p 0.154; Holm 0.46) | -1.37% (p 0.049; Holm 0.20) | -0.16% (p 0.383) |
| last hour, expiry days | -0.55% (p 0.350; Holm 1.00) | +0.37% (p 0.554; Holm 1.00) | -1.01% (p 0.253; Holm 1.00) | -0.59% (p 0.373; Holm 1.00) | -0.48% (p 0.290) |

Two cells in the robustness families pass Holm on all days, both in `hybridiv` at 15 minutes. First, US-signed GEX beats gross gamma by 0.37% (p 0.013, Holm 0.04, in `ablation_next15.csv`), but adding US-signed GEX to `hybridiv` is itself only -0.24% (p 0.106), because the gross set slightly hurts. Second, the real US set beats its placebo by 0.61% (Holm below 0.01), mostly because the placebo costs 0.36%. Neither is a gain over the model without GEX.

**Under MSE** (hybrid, `ablation_<target>_mse.csv`, raw p only):

| Target, days | gross (unsigned) | US sign | Pro sign | client mirror | placebo |
| --- | --- | --- | --- | --- | --- |
| 15 min, all days | +3.70% (p 0.829) | +3.25% (p 0.819) | +2.50% (p 0.818) | +3.79% (p 0.830) | +7.60% (p 0.845) |
| 15 min, expiry days | -2.47% (p 0.132) | -3.35% (p 0.149) | -2.73% (p 0.131) | -2.25% (p 0.123) | +1.06% (p 0.862) |
| 30 min, all days | -2.22% (p 0.123) | -2.54% (p 0.115) | -1.05% (p 0.098) | -0.43% (p 0.106) | -0.26% (p 0.108) |
| 30 min, expiry days | -5.81% (p 0.147) | -8.25% (p 0.156) | -5.55% (p 0.146) | -6.61% (p 0.152) | -4.08% (p 0.143) |
| last hour, all days | -0.98% (p 0.311) | -3.03% (p 0.033) | -3.98% (p 0.111) | -0.44% (p 0.382) | -1.54% (p 0.088) |
| last hour, expiry days | -4.09% (p 0.174) | -1.69% (p 0.108) | -4.69% (p 0.131) | -0.26% (p 0.389) | -0.70% (p 0.286) |

No GEX set improves MSE significantly at 15 or 30 minutes. In the last hour US-signed GEX has p 0.033 under MSE, but the placebo is at p 0.088 there too.

### 4.6 Model Confidence Set (REAL DATA)

At the 10% level on daily QLIKE (files `mcs_<target>_<subset>.csv`): at 15 and 30 minutes the set keeps 18 of 32 models on all days (every one a LightGBM-based model, plus one ridge model) and excludes mcsGARCH; on expiry days it keeps 25 of 32 and still excludes mcsGARCH and HAR. For the last hour the set is wide (27 of 32 on all days, 28 of 32 on expiry days, where it keeps mcsGARCH): the last hour has one forecast per day, so it separates models poorly. On daily MSE (`*_mse.csv`) the set keeps 30 to 32 of 32 models for every target, mcsGARCH included: MSE cannot separate the models in this sample.

### 4.7 Seed noise (REAL DATA)

Single-seed reruns of four LightGBM models at 30 minutes (`seed_noise_next30.csv`, seeds 7, 8 and 9; the final tables use the average of 5 seeds): the same hybrid's score against mcsGARCH was -7.46%, -7.83% and -7.63%, and a DM test between seeds 7 and 8 of this identical model gives p = 0.024 (plain LightGBM: 0.34 points, p 0.046). So a single seed moves a model by up to 0.4 points, and the DM test, which treats forecasts as fixed, can call that "significant". Averaging 5 seeds shrinks this noise, but GEX effects of 0.1 to 0.4 points are still of the same order.

## 5. Limitations

* **One market, four years.** 942 days, 792 out-of-sample test days, 169 expiry test days. The hold-out has 330 test days (70 expiry days), and the closing-auction period only 63 test days (13 expiry days). Small differences (below about 1 point of QLIKE) cannot be separated reliably, as Iteration 7 shows.
* **End-of-day positioning only.** GEX is built from the bhavcopy published at the previous close. Intraday changes in open interest, which matter most on expiry day, are not observed. A null result for end-of-day GEX is not a null result for intraday hedging flows.
* **The sign of dealer gamma is not observed.** NSE's participant file covers all index options together (not Nifty only) and is not broken down by strike, so the "participant" and "client" rules apply one weight to every strike. The US convention is an assumption about who sells calls and buys puts that may be wrong in India.
* **The DM test ignores estimator randomness.** It treats each model's forecasts as fixed. Five-seed averaging reduces the problem, and the seed study measures it, but a p-value near 0.05 for a sub-1-point gain should be read with that in mind.
* **QLIKE against MSE.** The conclusions are stated for QLIKE, the loss that ranks variance forecasts consistently and does not let a few huge windows dominate. Under MSE the ranking at 15 and 30 minutes is very different (Section 4.4), but the MSE differences are not significant and the MSE Model Confidence Set keeps almost every model, so MSE gives no reliable ranking in this sample. A user whose costs grow with the square of the error (for example, a hedger hurt mostly by the rare crash morning) should not read the QLIKE ranking as the answer to their question.
* **Multiple comparisons.** Holm's adjustment is applied within each family of the ablation (one model family, one target, one subset). It is not applied across targets, subsets or families, so the raw count of "p below 0.05" cells overstates the evidence.
* **Two protocols.** The development tables in Section 3 were made before the review changes of Iteration 10 (63-day mcsGARCH refit, normal-distribution p-values, a whole-sample order-flow scale, placebo days drawn from all days). Section 4 uses the final protocol throughout, including for the development window.
* **The hold-out is not unseen data.** The first real-data run (`results/pipeline_nse/`), whose results were known before this study began, covered the hold-out days too. The development runs never computed those days and no design choice here was made on them, but the hold-out is a check on the design choices, not data nobody had looked at.
* **EWMA fallback in two refits.** In 2 of the 38 refits (trained on the first 171 and 192 days; they forecast 42 test days from August to October 2023) the daily EGARCH inside mcsGARCH did not converge and `daily_egarch_variance` fell back to an EWMA daily variance (lambda 0.94), as designed. `mcsgarch_params.csv` records this in the column `daily_method`. It affects mcsGARCH and every model built on it in those weeks.
* **One development/hold-out split.** The choices in Section 3 were made on data up to 30 June 2025 only, but a single split is still one draw. The headline family (`hybrid`) was chosen on development data, which makes the hold-out a fair check of that choice, not an independent replication of every number.
* **Implied variance clock.** Implied variance is per trading day; weekends and holidays are not modelled separately, and the risk-free rate is a constant 6.5%. Settlement prices near expiry are noisy for strikes that barely traded.
* **Minute bars have no volume.** The order-flow feature is therefore a signed-return proxy, and the volume feature is constant.
* **The closing auction** appears only in the last 63 days, so the models see the `closing_auction` indicator in training only in the last two or three refits.

## 6. How to reproduce

```bash
pip install -e ".[dev]"
PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 python -m pytest -q

# the final run (99 minutes on one core; --verbose prints one line per walk-forward block)
python scripts/run_pipeline.py --source nse --raw data/raw --end 2026-11-02 --out results/pipeline_nse_v2 --n-seeds 5 --seed-study 3 --verbose
# (the run in this document was made without --end, when the data ended on 2 November 2026; scripts/reproduce_all.py passes --end 2026-11-02 so that later downloads do not change it)

# the development tables (needs the development runs, made with the data cut at 30 June 2025)
python experiments/v2_development_summary.py --dev-dir <dev runs> --baseline results/pipeline_nse --out results/pipeline_nse_v2/development
```

Files in `results/pipeline_nse_v2/`: `run_info.json` (configuration, versions, timing, protocol), `scoreboard_<target>.csv` (every model and subgroup: QLIKE, MSE and trimmed QLIKE, each also relative to mcsGARCH), `tests_<target>.csv` (each model against mcsGARCH, for QLIKE and MSE), `gain_by_anchor_<target>.csv` (each model against mcsGARCH at every anchor time), `ablation_<target>.csv` (every with/without comparison, raw and Holm p-values) and `ablation_<target>_mse.csv` (the same under MSE), `mcs_<target>_<subset>.csv` (and `_mse` versions), `seed_noise_next30.csv`, `feature_importance_<target>.csv`, `linear_coefficients_<target>.csv`, `hyperparameters_<target>.csv`, `mcsgarch_params.csv`, `daily_implied_panel.csv`, `daily_gex_panel.csv`, `daily_market.csv`, `forecasts_<target>.csv`, `figures/` (including `ablation_<target>_<family>.png`), and `development/` (the development-window tables quoted in Section 3).

Code: `src/niftygex/implied.py` (new), `features.py` (implied, calendar and new GEX blocks, placebo), `forecast.py` (implied and mcsgarch_iv models, ridge, seed averaging, in-window tuning, pooled training, 21-day mcsGARCH refit and the `mcsgarch_r63` comparison model), `evaluation.py` (subgroups including the 09:15 anchor and the auction period, ablation table, Holm, trimmed QLIKE, gain by anchor time, DM with a Student t), `data.py` (all expiries, futures turnover, Bank Nifty expiries, strike sorting), `volatility.py` (vectorised mcsGARCH forecast, identical to the loop to 1e-13). Tests: `tests/test_implied.py` (including the timing test: features of day t use only the chain published at the close of day t-1) and `tests/test_forecast_v2.py` (including a test that rebuilds the whole feature table without the last days and checks that no earlier feature changes).
