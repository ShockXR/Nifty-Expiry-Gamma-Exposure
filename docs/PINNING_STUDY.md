# Do Nifty and Bank Nifty "pin" to option strikes on expiry day?

**REAL DATA.** NSE F&O bhavcopies, NSE participant-wise open interest and NSE official index closes, 2 January 2023 to 2 November 2026, plus third-party 1-minute index bars checked against NSE. Every number in this document comes from a file written by `experiments/pinning_study.py` (folder `results/pinning/`, log `results/pinning_stdout.txt`). The file each table comes from is named under the table.

Written for Shreyan Thanki, to discuss with Prof. Piyush Pandey. No finance or statistics background is assumed; Section 3 explains every idea used.

## Findings first

1. **Pooled over all 200 Nifty expiry days, the evidence for pinning is suggestive but not conclusive.** On expiry days the official close was on average 11.52 index points from the nearest 50-point strike, against 12.62 points on the 739 other days (12.5 is what "no clustering at all" would give). The difference is 1.10 points (0.41 basis points of the index), 95% interval from 2.25 points closer to 0.03 points farther, permutation p = 0.054. This was our first pre-specified test (P1). It misses the usual 5% line.
2. **The pooled number hides a large change over time.** From 20 November 2024 to 31 August 2025 (one weekly expiry, on Thursday) the expiry-day close was 8.30 points from a strike against 12.82 on other days: 4.52 points closer (95% interval 2.12 to 6.86), p = 0.0006, and 39% of expiry closes fell within 5 points of a strike against 20% on other days. Since Nifty expiry moved to Tuesday (1 September 2025) there is no pinning at all (14.55 against 13.37 points, p = 0.27). The Thursday and Tuesday regimes differ with p = 0.0008. By half-year, expiry-day closes were closest to strikes in the second half of 2024 (6.6 points on average) and farthest in the first half of 2026 (17.2 points).
3. **Bank Nifty pinned while it had weekly options.** From January 2023 to 19 November 2024 (98 expiries) its expiry close was 20.26 points from the nearest 100-point strike against 24.52 on other days (25 means no clustering): 4.26 points closer (95% interval 0.99 to 7.36), 0.86 basis points, p = 0.008. The gap appears only in the official close; at 15:00 there is none. With monthly options only (24 expiries since November 2024) there is no detectable effect.
4. **We find no link between pinning and our estimates of dealer gamma.** Under none of the three ways of guessing who holds the options (US convention, NSE "Pro" traders, mirror image of NSE "Client" traders) is pinning stronger when dealers appear to be long gamma near the strike (Holm-adjusted p = 0.90, 0.52 and 0.91; pre-specified test P2). Under two of the three rules the difference even points the other way. The sample could have detected a difference of about 2.6 to 3.1 points between the groups with 80% probability, so a large effect of this kind is unlikely, but a small one cannot be ruled out. The sign of dealer gamma is **not observed** on NSE; this null may say more about our proxies than about dealers.
5. **Pinning to one particular strike is weak.** The close is not unusually often at the strike with the most open interest on expiry day. For the strike with the largest gamma exposure there is a small excess (5 percentage points more often within 10 points than on other days, p = 0.023), one of many exploratory tests.
6. **Two facts we verified from the data are useful in their own right.** On all 200 Nifty and 122 Bank Nifty expiry days, every expiring option's settlement price in the bhavcopy equals NSE's official index close exactly, so the official close is the price that matters. And since 3 August 2026 that close comes from a closing auction (NSE circulars, read directly), which changes the settlement window.

**In one sentence:** expiry-day pinning on NSE indices is real in some periods (Nifty in late 2024 to mid 2025, Bank Nifty while it had weekly options) and absent in others (Nifty since September 2025), and none of our public-data proxies for dealer gamma explains when it happens.

## 1. The question in plain words

**Options, strikes and expiry.** A Nifty call option with strike 24,000 pays the buyer the amount by which Nifty ends above 24,000 on the expiry day (a put pays the amount by which it ends below). NSE lists strikes every 50 points for Nifty and every 100 points for Bank Nifty near the current level. Index options are settled in cash at a single number: the index's official closing value on the expiry day.

**Pinning.** "Pinning" means the index tends to finish unusually close to a strike on expiry day. If the closing level were unrelated to the strike grid, the close would land anywhere between two strikes with equal chance, so its average distance to the nearest Nifty strike would be 12.5 points (a quarter of the 50-point gap). Pinning shows up as a smaller distance on expiry days.

**Why it might happen: dealer hedging.** Banks and trading firms that sell options ("dealers") usually hedge by trading the index or its futures. How their hedge changes when the index moves depends on gamma, the rate at which an option's sensitivity to the index changes. A dealer who is **long gamma** (has bought options, net) must sell when the index rises and buy when it falls to stay hedged. Near expiry, gamma is concentrated at strikes, so this selling above and buying below a heavily held strike pulls the index toward it. Avellaneda and Lipkin (2003) model exactly this. By the same logic, a dealer who is **short gamma** buys rises and sells falls, which pushes the price away from the strike. So hedging theory predicts pinning when dealers are long gamma near the strike.

**Evidence elsewhere.** Ni, Pearson and Poteshman (2005) found that US stock prices cluster at strikes on expiry dates and linked it to hedge rebalancing by option market makers and to trading that deliberately moves prices toward strikes. Golez and Jackwerth (2012) found pinning in S&P 500 index futures, the closest analogue to Nifty. Ni, Pearson, Poteshman and White (2021) showed more generally that market makers' net gamma affects how much US stocks move. Our literature review (`docs/LITERATURE_REVIEW.md`) found no peer-reviewed study of dealer hedging effects on NSE options, so this is a new question for India.

**What we ask.** (a) Does the official close cluster at strikes on expiry days more than on other days? (b) Does the index move toward strikes late in the day? (c) Is any clustering stronger when dealers are likely long gamma near the strike? (d) Does it differ across NSE's expiry regimes?

## 2. Data

| Data | Source | Period used | Used for |
| --- | --- | --- | --- |
| F&O bhavcopy, one file per trading day (944 files) | NSE archive | 2 Jan 2023 to 2 Nov 2026 | strike grid, open interest (OI), settlement prices, real expiry dates |
| Participant-wise open interest (Client, DII, FII, Pro) | NSE archive | same | two of the three dealer-sign rules |
| Official daily index close (Nifty 50, Nifty Bank) | NSE archive | same | the outcome (settlement price) |
| 1-minute index bars | third-party GitHub datasets, checked against NSE daily highs and lows | Nifty to 2 Nov 2026, Bank Nifty to 22 Apr 2026 | prices at 14:30 and 15:00, the intraday path |

The panel has 939 trading days per index (3 January 2023 to 27 October 2026: the first day has no previous chain, and days after the last expiry in the data have no "next expiry"). Expiry days are read from the bhavcopies themselves (a day is an expiry day if contracts expiring that day trade on it), which catches holiday shifts.

**Facts checked on the data** (file `results/pinning/data_checks.csv`, REAL DATA):

| Check | Nifty | Bank Nifty |
| --- | --- | --- |
| Expiry days found in the bhavcopies | 200 | 122 |
| Expiry days on which every expiring option's settlement price equals the official close | 200 | 122 |
| Expiry-eve chains whose strikes within 3% of the index are exactly one grid step apart | 200 of 200 (50 points) | 122 of 122 (100 points) |
| Days whose option data come strictly from the previous trading day (no look-ahead) | 939 of 939 | 939 of 939 |
| Days that needed a holiday re-dated contract (see Section 3.7) | 4 | 12 |

Nifty expiry days by regime and weekday: 98 in the multi-weekly regime (93 Thursdays, 5 Wednesdays), 41 in the single weekly Thursday regime (39 Thursdays, 2 Wednesdays), 61 in the Tuesday regime (56 Tuesdays, 5 Mondays).

**The settlement window, and what we assume.**

* Before 3 August 2026 we assume, as the project brief states, that NSE's official close is an average over the last 30 minutes of trading (15:00 to 15:30 IST). We could not open NSE's methodology page from this machine, but the data agree: the mean of the 15:00 to 15:29 one-minute bar closes is within 1.03 points of the official close on the median day and within 5 points on 96% of 878 Nifty days, while the last bar's close is 9.40 points away on the median day (`t4_close_proxy_check.csv`).
* From 3 August 2026 NSE runs a **closing auction session** for stocks that have derivatives, and their closing prices come from the auction (NSE circular NSE/CMTR/73362 of 18 March 2026, section 5.1.1; go-live date in circular NSE/CMTR/75479 of 30 July 2026; both read directly from nsearchives.nseindia.com). In the minute bars the index then stays frozen from 15:15 until about 15:28 and jumps to the official close. For these 13 Nifty expiry days we still use the official close (the settlement price is still equal to it), but we do not use bar prices inside the window.

## 3. Method in plain words

### 3.1 The main measure: distance to the nearest strike

For each day we take the official close and compute its distance to the nearest strike: for example, a close of 17,992.15 is 7.85 points from 18,000. For Nifty this distance is between 0 and 25 points; if closes ignore the strike grid it averages 12.5. For Bank Nifty it is between 0 and 50 and averages 25. We also express it in basis points (hundredths of a percent) of the index level.

Two things make this a fair test:

* **Non-expiry days are the control.** If round numbers attracted the index on every day, the non-expiry days would show it too. They do not: the non-expiry average is 12.62 for Nifty (p = 0.64 against 12.5) and 24.81 for Bank Nifty (p = 0.71 against 25) (`t1_clustering.csv`). So "no clustering" is a good description of ordinary days, and expiry days are compared with them.
* **Placebo strikes.** We repeat the count with fake strikes shifted by half a grid step (25 points for Nifty). If expiry closes really gather at strikes, they should gather less, not more, at the fake strikes.

### 3.2 Placebo days with the same structure

Besides all non-expiry days we use: the day before expiry for the same contract (paired by expiry week); the profile by trading days left to expiry (0, 1, 2, 3, 4); and a weekday swap (Thursdays when Thursday was the expiry day against Thursdays after expiry moved to Tuesday, and the reverse for Tuesdays). The weekday swap separates "expiry" from "day of the week".

### 3.3 Special strikes from the previous day's option chain

For each day we take the option chain of the contract that expires on the next expiry day **from the previous trading day's bhavcopy** (published after the previous close, so known before the day starts). From it we find the strike with the most open interest, the "max pain" strike (where option buyers would be paid the least) and the strike with the largest gamma exposure (computed with the package's Black-76 greeks from settlement prices). We then ask whether the close lands within 10 points of that strike more often than within 10 points of the two placebo strikes 25 points either side.

### 3.4 Movement toward the strike

A natural idea is to compare the distance to a strike before and after (for example at 15:00 and at the close). But under pure randomness the distance to a fixed point tends to grow, and grows more on volatile days (a consequence of Jensen's inequality). Expiry days are more volatile, so this measure is biased against finding pinning. We therefore use the **signed move toward the strike**: the index move from 14:30 (or 15:00) to the close, counted positive if it went toward the strike that was nearest at the start. With no pinning its average is zero whatever the volatility. We report both measures.

### 3.5 Dealer gamma, under three guesses

NSE does not publish who holds which option, so the sign of dealers' gamma is **not observed**. We guess it three ways (as everywhere in this project):

* **US convention:** dealers long all calls and short all puts.
* **Pro:** dealers hold the same net share of calls and of puts as NSE's "Pro" category does across all index options.
* **Client mirror:** dealers take the opposite side of NSE's "Client" category.

For a strike K we compute dealers' net gamma exposure as if the index were at K with 30 trading minutes left (the settlement window), using the previous day's open interest and implied volatilities. With so little time left this weights the strike itself and its immediate neighbours. The pre-specified test (P2) splits expiry days by the sign of this number at the strike nearest the close and compares the distance to that strike. Because the distance within a strike cell is spread evenly under no pinning whichever strike is nearest, both groups should average 12.5 if gamma does not matter. We also use the sign of the whole chain's net gamma at the previous day's forward price (known in advance) and an unsigned measure (how much of the chain's open interest sits at the nearest strike).

### 3.6 Statistics, in plain words

* **Unit of analysis: the day.** Every test treats one day as one observation; we never treat minutes as independent.
* **Permutation test.** To ask whether expiry days differ from other days, we shuffle the "expiry" labels across days 10,000 times and see how often a shuffled difference is as large as the real one. That share is the p-value: roughly, how surprising the result would be if expiry days were not special. Small p (below 0.05) means "hard to explain by chance".
* **Bootstrap interval.** We resample days with replacement 10,000 times; the middle 95% of the resampled differences is the 95% interval, a range of plausible effect sizes.
* **Several tests at once.** With many tests some small p-values appear by chance. For the pre-specified gamma test we apply a Holm correction over the three sign rules. Everything not marked as pre-specified is exploratory.
* **Power.** We simulate how large an effect our sample sizes could detect 80% of the time (Section 4.8).

**Pre-specified tests** (fixed in the script's docstring before any result was seen): P1, Nifty expiry against non-expiry days, mean distance to the nearest strike; P2, on Nifty expiry days, positive against negative dealer gamma near that strike, under each of the three rules with a Holm correction. The three-regime split was part of the brief. Everything else, including the half-year table and the regime comparison test, is exploratory.

### 3.7 Design changes made while working, and why

From `results/pinning/design_log.csv`:

| Change | Reason |
| --- | --- |
| Use NSE's official close as the outcome, not the mean of the 15:00 to 15:29 bars | The official close is exactly the settlement price on every expiry day; the bar mean is about 1 point away on a typical day |
| Define the "max gamma" strike from unsigned gamma exposure | A signed version depends on the unobserved sign; the sign enters only in the conditioning step |
| Use the signed move toward the strike as the main movement measure | The change in distance is biased upward on volatile days (Section 3.4) |
| Sort the chain by strike before filling missing implied volatilities | Newer bhavcopy files are not in strike order (the loader now sorts too) |
| Use the contract re-dated by a holiday when the expiry date in the previous day's file differs by up to 3 days | On 27 June 2023 the Nifty contract was still listed for 29 June but expired on 28 June; similar re-dating for Bank Nifty in September 2023 and January 2025 |
| Split the Tuesday regime at 3 August 2026 and keep bar prices inside the window to earlier days | NSE's closing auction started that day (Section 2) |
| Add a test of whether regimes differ, and a half-year table | The first run showed opposite signs in the Thursday and Tuesday regimes; whether they differ must be tested, not read off two separate p-values |
| Gamma "near a strike" evaluated with 30 minutes left | With a full day left, gamma spreads over about five strikes either side |

## 4. Results (all REAL DATA)

### 4.1 Clustering at strikes: expiry days against other days

Mean distance from the official close to the nearest strike (points). "Within 5" is the share of closes within one tenth of a grid step of a strike (5 points for Nifty, 10 for Bank Nifty); with no clustering it is 20%.

| Index and sample | Expiry days | Mean, expiry | Mean, other days | Difference (95% interval) | In basis points | p | Within 5, expiry | Within 5, other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nifty, all (P1) | 200 | 11.52 | 12.62 | -1.10 (-2.25 to 0.03) | -0.41 | 0.054 | 25.0% | 20.8% |
| Nifty, multi-weekly (to 19 Nov 2024) | 98 | 10.99 | 12.07 | -1.09 (-2.70 to 0.47) | -0.36 | 0.18 | 27.6% | 22.0% |
| Nifty, single weekly Thursday (21 Nov 2024 to 29 Aug 2025) | 41 | 8.30 | 12.82 | -4.52 (-6.86 to -2.12) | -1.89 | 0.0006 | 39.0% | 19.9% |
| Nifty, Tuesday (from 1 Sep 2025) | 61 | 14.55 | 13.37 | 1.18 (-0.74 to 3.06) | 0.50 | 0.27 | 11.5% | 19.6% |
| of which before the closing auction | 48 | 14.52 | 13.28 | 1.25 (-0.94 to 3.41) | 0.52 | 0.29 | 10.4% | 18.5% |
| of which closing auction (from 3 Aug 2026) | 13 | 14.66 | 13.75 | 0.91 (-3.67 to 5.13) | 0.42 | 0.71 | 15.4% | 23.9% |
| Nifty, monthly expiries only | 46 | 12.37 | 12.62 | -0.26 (-2.44 to 1.90) | -0.01 | 0.82 | 21.7% | 20.8% |
| Nifty, weekly (non-monthly) expiries | 154 | 11.27 | 12.62 | -1.35 (-2.59 to -0.08) | -0.53 | 0.037 | 26.0% | 20.8% |
| Bank Nifty, all | 122 | 21.01 | 24.81 | -3.80 (-6.54 to -1.01) | -0.45 | 0.007 | 27.9% | 19.3% |
| Bank Nifty, weekly options (to 19 Nov 2024) | 98 | 20.26 | 24.52 | -4.26 (-7.36 to -0.99) | -0.86 | 0.008 | 28.6% | 17.9% |
| Bank Nifty, monthly only (from 21 Nov 2024) | 24 | 24.07 | 25.05 | -0.98 (-7.34 to 5.39) | -0.16 | 0.75 | 25.0% | 20.5% |

Source: `t1_clustering.csv`. Bank Nifty distances are on a 100-point grid (no clustering = 25).

The placebo strikes behave as expected: on all Nifty expiry days 16.5% of closes are within 5 points of a half-way point, against 25.0% within 5 points of a real strike; on other days the two shares are 19.8% and 20.8% (`t1_clustering.csv`). Within 2.5 points of a strike: 14.0% of Nifty expiry closes against 8.9% of other days (10% with no clustering).

Figure: `results/pinning/fig_close_offset_histogram.png` shows where the Nifty close lands inside its strike cell on expiry and other days.

### 4.2 Do the regimes really differ?

Each "gap" is expiry minus other days within a regime. The test asks whether two gaps are equal (`t1_regime_heterogeneity.csv`).

| Comparison | Gap A | Gap B | A minus B (95% interval) | p |
| --- | --- | --- | --- | --- |
| Nifty multi-weekly (A) vs single weekly Thursday (B) | -1.09 | -4.52 | 3.43 (0.53 to 6.35) | 0.023 |
| Nifty multi-weekly (A) vs Tuesday (B) | -1.09 | 1.18 | -2.27 (-4.69 to 0.23) | 0.091 |
| Nifty single weekly Thursday (A) vs Tuesday (B) | -4.52 | 1.18 | -5.70 (-8.77 to -2.57) | 0.0008 |
| Bank Nifty weekly options (A) vs monthly only (B) | -4.26 | -0.98 | -3.28 (-10.32 to 3.60) | 0.36 |

### 4.3 Over time (exploratory, no tests)

Mean distance on Nifty expiry days by half-year, with its standard error, and the non-expiry mean (`t8_by_half_year.csv`; figure `fig_by_half_year.png`):

| Half-year | Expiry days | Mean, expiry (standard error) | Mean, other days | Expiry closes within 5 points |
| --- | --- | --- | --- | --- |
| 2023 H1 | 26 | 13.80 (1.33) | 11.08 | 7.7% |
| 2023 H2 | 26 | 11.71 (1.49) | 12.39 | 26.9% |
| 2024 H1 | 26 | 10.27 (1.37) | 12.24 | 34.6% |
| 2024 H2 | 26 | 6.62 (1.24) | 12.91 | 50.0% |
| 2025 H1 | 26 | 8.55 (1.37) | 12.14 | 38.5% |
| 2025 H2 | 27 | 11.52 (1.17) | 14.00 | 14.8% |
| 2026 H1 | 26 | 17.23 (1.12) | 12.67 | 3.8% |
| 2026 H2 (to 27 Oct) | 17 | 12.99 (1.86) | 13.88 | 23.5% |

Clustering builds through 2024, peaks in the second half of 2024, fades in the second half of 2025 and turns into the opposite (closes far from strikes) in the first half of 2026. This pattern was seen after looking at the data. It should be treated as a description to be checked on future data, not as a tested result.

### 4.4 Days to expiry

Mean distance by trading days left to the next expiry (`t2_days_to_expiry_profile.csv`):

| Days to expiry | Nifty days | Nifty mean (95% interval) | Bank Nifty days | Bank Nifty mean (95% interval) |
| --- | --- | --- | --- | --- |
| 0 (expiry day) | 200 | 11.52 (10.54 to 12.54) | 122 | 21.01 (18.39 to 23.68) |
| 1 | 200 | 12.74 (11.69 to 13.74) | 122 | 24.89 (22.35 to 27.43) |
| 2 | 200 | 12.44 (11.45 to 13.45) | 122 | 24.87 (22.43 to 27.32) |
| 3 | 196 | 13.02 (12.06 to 13.99) | 118 | 24.11 (21.52 to 26.74) |
| 4 | 143 | 12.15 (10.96 to 13.33) | 93 | 24.61 (21.75 to 27.43) |

Only the expiry day itself stands out; the day before expiry looks like any other day.

### 4.5 When in the afternoon does it appear?

Expiry minus other days, mean distance to the nearest strike of each price (`t4_intraday_movement.csv`, days before the closing auction; Nifty 187 expiry and 691 other days, Bank Nifty 115 and 694):

| Price | Nifty difference (p) | Bank Nifty difference (p) |
| --- | --- | --- |
| at 14:30 | -0.64 (0.27) | -1.58 (0.27) |
| at 15:00 | -1.08 (0.073) | 0.96 (0.50) |
| last 1-minute bar (15:29) | -0.86 (0.15) | -2.40 (0.096) |
| mean of 15:00 to 15:29 bars | -1.07 (0.071) | -2.72 (0.063) |
| official close | -1.25 (0.034) | -3.99 (0.007) |

By regime (`t4_intraday_by_regime.csv`): in the Nifty Thursday single-weekly regime the gap is small at 14:30 (-0.80 points, p = 0.52), already large at 15:00 (-3.70, p = 0.003) and largest at the official close (-4.52, p = 0.0004). For Bank Nifty with weekly options the gap is absent at 14:30 and 15:00 (-1.11 and 0.70 points) and appears only in the official close (-4.25, p = 0.007). So for Bank Nifty the clustering forms inside the settlement window itself; for Nifty in its strongest period it forms between 14:30 and 15:00 and grows a little into the close. Figure `fig_intraday_distance.png` shows the Nifty path minute by minute.

**Signed move toward the strike** (`t4_intraday_movement.csv`, all days with bars):

| Start | Index | Expiry days: mean move toward (points) | Other days | Difference (95% interval) | p |
| --- | --- | --- | --- | --- | --- |
| 14:30 | Nifty | 5.93 | -1.16 | 7.09 (-1.48 to 15.74) | 0.066 |
| 15:00 | Nifty | 0.63 | 0.38 | 0.25 (-5.15 to 5.62) | 0.93 |
| 14:30 | Bank Nifty | 10.28 | -2.86 | 13.14 (-12.09 to 38.25) | 0.30 |
| 15:00 | Bank Nifty | 10.16 | 0.37 | 9.78 (-5.40 to 24.93) | 0.20 |

The moves point toward strikes on expiry days, but they are noisy (the average absolute move from 15:00 to the close is about 23 to 24 Nifty points) and none is significant.

### 4.6 Special strikes from the previous day's chain

"Excess hit" is the share of closes within 10 points (20 for Bank Nifty) of the strike minus the same share for the two placebo strikes 25 (50) points away (`t3_target_strikes.csv`). Expiry days against all other days:

| Strike | Nifty: expiry excess hit | Nifty: expiry minus other days (p) | Bank Nifty: expiry excess hit | Bank Nifty: expiry minus other days (p) |
| --- | --- | --- | --- | --- |
| Most open interest | 0.005 | 0.017 (0.33) | 0.012 | 0.015 (0.42) |
| Max pain | 0.018 | 0.028 (0.18) | 0.025 | 0.041 (0.082) |
| Largest gamma exposure | 0.040 | 0.051 (0.023) | 0.045 | 0.046 (0.041) |

These strikes are usually far from where the index closes (on Nifty expiry days the most-OI strike was on average 320 points from the close), so they are rarely hit at all: 3.0% of Nifty expiry closes were within 10 points of the most-OI strike. The small excess at the largest-gamma strike is consistent with the general clustering in 4.1, and it is one of many exploratory tests.

### 4.7 Dealer gamma (pre-specified test P2 and companions)

Nifty expiry days, mean distance to the nearest strike, split by the sign of dealer gamma (`t5_gamma_conditioning.csv`). Theory predicts the "positive" group is closer (a negative difference).

| What is split | Rule | Positive: days, mean | Negative: days, mean | Positive minus negative (95% interval) | p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| Gamma near the nearest strike (P2) | US convention | 97, 11.92 | 103, 11.14 | 0.78 (-1.20 to 2.84) | 0.45 | 0.90 |
| Gamma near the nearest strike (P2) | Pro | 101, 12.21 | 99, 10.82 | 1.40 (-0.65 to 3.41) | 0.17 | 0.52 |
| Gamma near the nearest strike (P2) | Client mirror | 181, 11.50 | 19, 11.70 | -0.19 (-4.09 to 3.73) | 0.91 | 0.91 |
| Whole chain at the forward | US convention | 175, 11.32 | 25, 12.93 | -1.61 (-4.52 to 1.49) | 0.32 | |
| Whole chain at the forward | Pro | 91, 12.23 | 109, 10.93 | 1.30 (-0.68 to 3.34) | 0.21 | |
| Whole chain at the forward | Client mirror | 189, 11.74 | 11, 7.83 | 3.91 (-0.71 to 8.00) | 0.085 | |
| Share of OI at the nearest strike above its median (unsigned) | none | 100, 12.21 | 100, 10.83 | 1.38 (-0.57 to 3.43) | 0.19 | |

The rules disagree with each other about the sign most of the time: near the strike, the US and Pro rules agree on only 46% of Nifty expiry days, and the client-mirror rule calls dealers long gamma on 90.5% of them (`t5_sign_agreement.csv`). For Bank Nifty no expiry-day split is significant either (Holm p 0.87 to 1.00). Seven of the 28 Bank Nifty rows in that table have p below 0.05: five are on placebo (non-expiry) days and two are difference-in-differences rows driven by those five; six of the seven use the client-mirror rule, where only 10 to 32 days fall in the negative group. Since they appear on days when theory predicts nothing and in very small groups, we read them as a warning about small groups and many tests, not as a finding.

### 4.8 Power: how big an effect could we have seen?

Simulated with the real sample sizes, two-sided 5% tests (`t6_power_mde.csv`, `t6_power.csv`). "Pinned share" means a fraction of expiry closes sit within 2.5 points of a strike; "pull" means every expiry close is moved part of the way to its strike.

| Comparison | Days (group 1, group 0) | 80% power at | Equivalent shift in mean distance |
| --- | --- | --- | --- |
| Nifty expiry vs other days (P1) | 200, 739 | 15.5% of closes pinned, or a 12.4% pull | about 1.6 to 1.7 points |
| Nifty P2, US convention | 97, 103 | 27.6% pinned, or a 20.9% pull | about 2.6 to 3.1 points |
| Nifty P2, Pro | 101, 99 | 27.3% pinned, or a 20.8% pull | about 2.6 to 3.1 points |
| Nifty P2, Client mirror | 181, 19 | 46.5% pinned, or a 40.3% pull | about 5.0 to 5.2 points |
| Nifty Thursday single-weekly regime | 41, 151 | 35.9% pinned, or a 23.9% pull | about 3.0 to 4.1 points |

The observed pooled effect (1.10 points) is below what P1 detects reliably, which is why a real but small pooled effect can still give p = 0.054. The Thursday-regime effect (4.52 points) is well above its detectable size.

## 5. Robustness

From `t7_robustness.csv` and `t1_clustering.csv`:

* **Same weekday, different status.** Thursday expiries are 10.16 points from a strike against 13.55 on Thursdays after expiry moved to Tuesday (difference -3.40, 95% interval -5.60 to -1.18, p = 0.004). Tuesday expiries are 14.79 against 12.53 on Tuesdays that were not Nifty expiries (difference 2.26, interval 0.11 to 4.40, p = 0.039). So the Thursday effect is not a Thursday effect, and the Tuesday regime shows, if anything, closes pushed away from strikes.
* **Shuffling only within weekdays** weakens the pooled Nifty result (p = 0.19 instead of 0.054), because the expiry effect has opposite signs on Thursdays and Tuesdays. For Bank Nifty with weekly options it stays (p = 0.011).
* **Round hundreds.** Distance to the nearest multiple of 100 shows no expiry effect for Nifty (-1.41 points, p = 0.22), so the clustering is at the 50-point grid, not at round numbers only.
* **Holiday-shifted expiries removed.** Keeping only expiries on the scheduled weekday: -1.09 points, p = 0.071.
* **Monthly against weekly.** Monthly Nifty expiries show no effect (-0.26 points, p = 0.82); weekly ones do (-1.35, p = 0.037).
* **Share within 5 points** instead of the mean: Thursday single-weekly regime 39.0% against 19.9%, p = 0.013; pooled Nifty 25.0% against 20.8%, p = 0.22.

## 6. Limitations

* **The dealer sign is not observed.** NSE's participant file is not by strike, not by expiry and covers all index options together. Our three rules disagree with each other on most days. The P2 null therefore rules out a strong relationship with these proxies, not a relationship with true dealer positions.
* **The time pattern is exploratory.** We split by the three regimes because the brief asked for it; the striking rise and fall by half-year was seen after looking at the data. Several things changed around mid-2025 at once (the move to Tuesday on 1 September 2025, other SEBI measures from November 2024, and SEBI's July 2025 interim order on alleged index manipulation around expiry days, see below), and our data cannot tell them apart.
* **Hedging is only one possible cause.** Ni, Pearson and Poteshman (2005) also link US clustering to trading that deliberately moves prices toward strikes. In India, SEBI's interim order of July 2025 in the matter of index manipulation by Jane Street Group (reference WTM/AN/MRD/MRD-SEC-3/31516/2025-26) makes allegations about trading near the close on index expiry days. It is an interim order, the firm disputes it, and we could not open the full text from this machine (sebi.gov.in was not reachable), so we only note it as context and draw no conclusion from it.
* **Small samples in the newest regimes.** 41 Thursday single-weekly expiries, 61 Tuesday expiries, 13 under the closing auction, 24 monthly Bank Nifty expiries.
* **Minute bars are third-party.** They match NSE's daily highs and lows, but intraday results rest on them. Bank Nifty bars end on 22 April 2026.
* **The official close before August 2026 is assumed to be a 30-minute average.** The data support it (Section 2), but we did not read NSE's own methodology document.
* **Many exploratory tests.** Tables 4.5 to 4.7 contain dozens of p-values; a few below 0.05 are expected by chance.

## 7. What would make it stronger

1. **Real dealer positions.** Strike-level open interest by participant category, or an exchange data request for market-maker accounts (the US studies use such data), would replace guessed signs.
2. **Intraday option data on expiry day.** Open interest and trading during the day would show whether hedgers were long or short gamma at the strike in the final hour, which end-of-previous-day data cannot.
3. **More expiry days.** Each further year adds about 52 Nifty expiries; the half-year pattern should be re-tested on data after October 2026, especially under the closing auction.
4. **Other NSE and BSE indices.** Fin Nifty, Midcap Nifty and Sensex expiries give independent samples with different regimes and expiry days.
5. **Constituent stocks.** Since the official close is built from stock closing prices, looking at which stocks moved in the settlement window on pinned days would show how the index got there.

## 8. Files and how to rerun

```bash
OMP_NUM_THREADS=1 python experiments/pinning_study.py --raw data/raw --out results/pinning
OMP_NUM_THREADS=1 python -m pytest -q tests/test_pinning.py
```

The run is deterministic (fixed random seed) and takes about two minutes. Outputs in `results/pinning/`: `day_panel_nifty.csv` and `day_panel_banknifty.csv` (one row per day with every measure), `data_checks.csv`, `t1_clustering.csv`, `t1_regime_heterogeneity.csv`, `t2_days_to_expiry_profile.csv`, `t3_target_strikes.csv`, `t4_intraday_movement.csv`, `t4_intraday_by_regime.csv`, `t4_close_proxy_check.csv`, `t5_gamma_conditioning.csv`, `t5_sign_agreement.csv`, `t6_power.csv`, `t6_power_mde.csv`, `t7_robustness.csv`, `t8_by_half_year.csv`, `design_log.csv`, and three figures. The full log is `results/pinning_stdout.txt`. This study supersedes the quick check R4 in `experiments/real_day_level.py`.

## References

From `docs/LITERATURE_REVIEW.md` (verified there):

* Avellaneda, M. and Lipkin, M. D. (2003). A market-induced mechanism for stock pinning. *Quantitative Finance*, 3(6), 417 to 425. doi:10.1088/1469-7688/3/6/301
* Golez, B. and Jackwerth, J. C. (2012). Pinning in the S&P 500 futures. *Journal of Financial Economics*, 106(3), 566 to 585. doi:10.1016/j.jfineco.2012.06.010
* Ni, S. X., Pearson, N. D. and Poteshman, A. M. (2005). Stock price clustering on option expiration dates. *Journal of Financial Economics*, 78(1), 49 to 87. doi:10.1016/j.jfineco.2004.08.005
* Ni, S. X., Pearson, N. D., Poteshman, A. M. and White, J. (2021). Does option trading have a pervasive impact on underlying stock prices? *Review of Financial Studies*, 34(4), 1952 to 1986. doi:10.1093/rfs/hhaa082
* SEBI (2024). Measures to strengthen equity index derivatives framework. Circular SEBI/HO/MRD/TPD-1/P/CIR/2024/132, 1 October 2024.
* NSE (2025). Circular No. 108/2025 (NSE/FAOP/68685), 23 June 2025 (Nifty weekly expiry moved to Tuesday).

Checked for this study:

* NSE (2026). Operational guidelines and Standard Operating Procedure (SOP) on Introduction of Closing Auction Session (CAS) in the Equity Cash Segment. Circular NSE/CMTR/73362 (Circular Ref. No. 38/2026), 18 March 2026. https://nsearchives.nseindia.com/content/circulars/CMTR73362.pdf (read in full).
* NSE (2026). Mock trading on Saturday, August 01, 2026, Go Live of Trading modalities related to Introduction of Closing Auction Session (CAS) in Equity Cash segment. Circular NSE/CMTR/75479 (Circular Ref. No. 108/2026), 30 July 2026. https://nsearchives.nseindia.com/content/circulars/CMTR75479.pdf (read in full; states that CAS is live from 3 August 2026).
* SEBI (2025). Interim Order in the matter of Index manipulation by Jane Street Group, WTM/AN/MRD/MRD-SEC-3/31516/2025-26, July 2025. https://www.sebi.gov.in/sebi_data/attachdocs/jul-2025/1751584518593.pdf. Partly verified: title, reference number and address confirmed through a search of sebi.gov.in; the full text could not be opened from this machine.
