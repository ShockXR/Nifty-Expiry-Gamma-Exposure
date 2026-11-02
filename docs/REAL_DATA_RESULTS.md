# Real-data results

**Everything in this document comes from REAL market data:** NSE's own files (F&O bhavcopy, participant-wise open interest, daily index values including India VIX), downloaded from NSE's archive server, plus third-party 1-minute Nifty 50 bars from a public GitHub dataset that we checked day by day against NSE's official values. Simulation results are in `docs/EXPERIMENT_REPORT.md` (E1 to E7) and are kept separate.

Sample: 942 trading days, 2 January 2023 to 2 November 2026, of which 200 are Nifty expiry days. Out-of-sample forecasts start after a 150-day training period, so 792 test days. Every number below is copied from a file in `results/`; the file is named next to each table. (The tables were regenerated after a bug fix on 2 November, 20:45 UTC: UDiFF bhavcopies are not sorted by strike, which had affected how missing implied vols were filled. Net GEX changed noticeably on about 100 days where it is close to zero; no conclusion changed, and the largest change in any scoreboard cell was 1.4 points.)

| ID | Question | Command | Output |
| --- | --- | --- | --- |
| R0 | Are the data right? | `python scripts/validate_minute_bars.py`, `python scripts/check_oi_units.py` | `results/data_validation/` |
| R1 | Out of sample, which model forecasts Nifty's intraday volatility best, and do dealer-gamma (GEX) features help? | `python scripts/run_pipeline.py --source nse` | `results/pipeline_nse/` |
| R2 | How much extra volatility does an expiry day carry, and did that change under the later expiry rules? | `python experiments/real_day_level.py` | `results/real_day_level/` |
| R3 | In-sample, is yesterday's GEX related to today's volatility? | same | same |
| R4 | A first, crude look at pinning | same | same |
| R5 | Does the market's move over the day carry on into the last 30 minutes, and does that depend on dealer gamma? | `python experiments/momentum_gex.py` | `results/momentum_gex/` |
| R0b | Do the results survive NSE's closing auction (from 3 August 2026)? | `python experiments/closing_auction_check.py` | `results/closing_auction/` |

Separate real-data studies, each with its own document:

* `docs/REPLICATION_NSE.md`: the paper's Tables 1 to 6 rebuilt on NSE data. The "after" sample matches closely; the stated "before" window does not, and a 64-day window from 14 November 2022 to 10 February 2023 reproduces the printed "before" numbers (our main question for the authors).
* `docs/PINNING_STUDY.md`: a full pinning study (replaces the crude R4 check below).
* Still running: an event study of the expiry rule changes with Bank Nifty as a comparison, a stronger version of R1, and a falsification suite for R3 and R5.

---

## Findings (short)

1. **The data hold up.** Minute-bar highs and lows match NSE's official values within 0.05% on 943 of 943 days (highs) and 942 of 943 (lows) for Nifty. Bhavcopy open interest is in shares and adds up exactly to NSE's participant totals once contracts expiring that day are left out.
2. **The paper's model can be beaten out of sample.** A hybrid that keeps mcsGARCH (the paper's model) and corrects it with a LightGBM model of its errors forecasts the next 15 and 30 minutes about 7% and 9% better (QLIKE loss), on all days and on expiry days, with Diebold-Mariano p-values below 0.001. mcsGARCH is excluded from the Model Confidence Set for these horizons.
3. **For the last hour of the day, no model reliably beats mcsGARCH.** Over all test days HAR looks about 11% better (one-sided p = 0.015), but most of that edge comes from the last 63 days, after NSE introduced a closing auction on 3 August 2026 (see R0). Before that date HAR is about 4% better and the difference is not significant (p = 0.11). The machine-learning models add nothing for the last hour.
4. **GEX features do not improve out-of-sample forecasts** under any of the three sign rules. This matches what our simulations (E4) said to expect when the sign of dealer positions is uncertain and the effect is small.
5. **But GEX is related to volatility in-sample, with a sign that flips on expiry day.** On ordinary days, higher dealer gamma the evening before goes with lower volatility the next day, as hedging theory predicts. On expiry days the relation reverses. This holds under all three sign rules and after controlling for the expiry regime, but it is an in-sample association on a proxied sign, not a causal result (details in R3).
6. **Expiry days are more volatile than the rest of their week, but not in every regime.** About 11% more realized variance before November 2024, no clear difference from November 2024 to August 2025 (Thursday expiry, one weekly per exchange), and about 21% more since Nifty moved to Tuesday, concentrated in the last hour (about 43% more).
7. **Nifty reverses into the close rather than continuing, and the reversal is stronger on days when the Pro-based GEX is negative.** This is the opposite of what the US literature on hedging-driven momentum predicts, and it partly survives controls for VIX and the size of the day's move (R5). It does not fit neatly with finding 5 either, so the sign proxy itself is the open question.

---

## R0. Data checks

**Minute bars vs NSE** (`results/data_validation/summary.csv`). For each day we compare the highest 1-minute high, lowest low and first open with NSE's official values.

| | Nifty 50 | Nifty Bank |
| --- | --- | --- |
| Days compared | 943 | 811 (bars end 22 Apr 2026) |
| High within 5 basis points | 100% | 99.4% |
| Low within 5 basis points | 99.9% | 99.3% |
| Open within 5 basis points | 99.8% | 99.9% |
| Median absolute difference, high and low | 0 | 0 |
| Median absolute difference, close | 3.8 bp | 5.3 bp |
| Largest high or low difference | 6.8 bp | 23.3 bp |

Closes are not expected to match exactly: NSE's official close is an average over the last 30 minutes, while the last bar's close is the last trade. One official trading day has no bars (the 1 November 2024 Muhurat session) and five bar days are not in NSE's weekday files (Saturday or Sunday special sessions, such as budget days); both are left out of the analysis.

**NSE's closing auction** (`results/closing_auction/auction_evidence.csv`). From 3 August 2026 NSE sets the official close in a closing auction. In the minute bars the index stands still from 15:16 to 15:28 on 96% of those minutes and then jumps to the auction price in the 15:29 bar (average jump 8.3 basis points, against 2.1 from January to July 2026), and the 15:29 close equals NSE's official close on all 63 days (median difference 0, against about 5 basis points before). So for the last 63 days of the sample, "the last hour" and "the last 30 minutes" contain 13 dead minutes and one auction jump. Every result below was rerun without those days (`experiments/closing_auction_check.py`); where it matters, both versions are shown.

**Open interest units** (`results/data_validation/oi_units_check.csv`). Sum over all index options of OI divided by each row's lot size, compared with the participant file's TOTAL, on 9 dates: equal to within one contract on every date once contracts expiring that day are excluded. So OI is in shares, and the bhavcopy keeps expiring contracts' OI on their last day while the participant file does not.

---

## R1. Out-of-sample forecasting contest

Setup: forecasts of realized variance (sum of squared 1-minute returns) over the next 15 minutes and next 30 minutes from every 15-minute mark, and over the last hour from 14:30. Every model is refitted every 21 trading days on all data before the forecast period (expanding window), so no model ever sees the future. Loss: QLIKE (lower is better), shown as % difference from mcsGARCH. Files: `results/pipeline_nse/scoreboard_*.csv`, `tests_*.csv`, `mcs_*.csv`.

| Model | Next 15 min, all days | Next 15 min, expiry days | Next 30 min, all days | Next 30 min, expiry days | Last hour, all days | Last hour, expiry days |
| --- | --- | --- | --- | --- | --- | --- |
| mcsGARCH (the paper's model) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| HAR | +5.2 | +4.0 | +6.7 | +5.0 | **-10.8** | **-10.2** |
| India VIX only | +52.5 | +33.8 | +63.6 | +42.2 | +17.6 | +27.7 |
| LightGBM, no GEX | -3.9 | -6.2 | -4.2 | -6.6 | -0.5 | -8.2 |
| LightGBM, with GEX | -4.4 | -7.2 | -4.7 | -7.1 | +1.9 | -8.8 |
| Hybrid, no GEX | -6.7 | -6.2 | -9.0 | -6.6 | -1.3 | +3.2 |
| Hybrid, GEX (US sign) | -6.7 | -6.3 | -9.2 | -5.8 | -1.6 | +3.4 |
| Hybrid, GEX (Pro participant sign) | -6.8 | -6.5 | -9.2 | -6.1 | -1.3 | +2.5 |

Test anchors: 19,800 (next 15), 19,008 (next 30), 792 (last hour); expiry days are 169 of the 792 test days.

**Is the improvement real?** Diebold-Mariano tests on daily average losses (`tests_*.csv`; one-sided, asking whether the named model has lower loss than mcsGARCH, with the Harvey, Leybourne and Newbold small-sample correction):

* Hybrid (no GEX) vs mcsGARCH: p < 0.0001 at 15 and 30 minutes on all days, p = 0.0007 and 0.0002 on expiry days. Not significant for the last hour (p = 0.26).
* LightGBM (no GEX) vs mcsGARCH on expiry days: p = 0.001 (15 min) and 0.0001 (30 min).
* HAR vs mcsGARCH, last hour: p = 0.015 (all days), 0.07 (expiry days). Without the 63 closing-auction days: HAR is 4.2% better on all days (p = 0.11) and 7.1% on expiry days (p = 0.17), so the last-hour edge is not established (`results/closing_auction/r1_dm_tests.csv`).
* Without the closing-auction days the hybrid's gain at 15 and 30 minutes is about the same: 6.0% and 7.8% on all days, 5.5% and 5.7% on expiry days, all with p of 0.0013 or smaller.
* The Model Confidence Set (90%) excludes mcsGARCH, HAR, VIX and the naive models at 15 and 30 minutes; for the last hour it keeps almost every model, so the data cannot separate them there.

**Does GEX help?** The clean test compares the same model with and without the GEX block. For the hybrid, the extra loss reduction from GEX is zero to within noise under every sign rule (one-sided Diebold-Mariano p between 0.11 and 0.87 across horizons and day types). For LightGBM, GEX is about half a point better at 15 and 30 minutes, which is not significant (p = 0.13 and 0.09) and reverses in the last hour. We read this as no reliable out-of-sample value from end-of-day GEX.

**What the mcsGARCH fit looks like on the whole sample** (`mcsgarch_full_sample.csv`): intraday GARCH $\alpha = 0.049$, $\beta = 0.945$, so persistence 0.994 and an implied half-life of about 2 hours; Student-$t$ degrees of freedom 5.9. Across the 13 walk-forward refits $\alpha$ ranges from 0.025 to 0.051 (`mcsgarch_params.csv`).

**Caveats.** Minute volume does not exist for the index before 2026, so the volume feature is constant and the "order-flow" feature `ofi15` reduces to a price-based measure (the signed size of recent returns), not true order flow. The GEX features are built from end-of-day chains only.

---

## R2. Expiry-day volatility by expiry regime

For each week we take the expiry day's log realized variance minus the average log realized variance of the other days of that week (this cancels market-wide shocks that last the whole week). File: `results/real_day_level/r2_expiry_effect_by_regime.csv`.

| Regime | Weeks | Whole day: expiry vs rest of week | 95% interval | p (signed-rank) | Last hour: expiry vs rest of week | p |
| --- | --- | --- | --- | --- | --- | --- |
| Several weeklies, Thursday (to 19 Nov 2024) | 98 | +11% | +1% to +22% | 0.045 | +16% | 0.053 |
| One weekly per exchange, Thursday (20 Nov 2024 to 31 Aug 2025) | 41 | -11% | -26% to +8% | 0.26 | +10% | 0.65 |
| Nifty expiry on Tuesday (from 1 Sep 2025) | 61 | +21% | +8% to +38% | 0.005 | +43% | 0.00005 |

(Percentages are $e^{\text{mean}} - 1$ of the log differences.) The Tuesday result does not come from the closing auction: on the 48 Tuesday expiry weeks before 3 August 2026 the whole-day effect is +24% (p = 0.005) and the last-hour effect +42% (p = 0.00004) (`results/closing_auction/r2_tuesday_split.csv`). The Tuesday period differs from the Thursday single-weekly period (Mann-Whitney p = 0.012 for the whole day). Each period bundles several changes (SEBI's November 2024 package also raised contract sizes and added margin on expiry day), so these are descriptive, and three regimes times three measures is nine tests. A fuller event study with Bank Nifty as a comparison index is in progress.

---

## R3. Is yesterday's GEX related to today's volatility? (in-sample)

Regression of the log of today's realized variance on yesterday's log realized variance, its 5-day average, the log of yesterday's VIX, an expiry-day dummy, yesterday's net GEX (divided by its 20-day median gross GEX, so it is a share between about -0.3 and +0.6) and GEX times the expiry dummy. Newey-West standard errors with 5 lags, 937 days. File: `results/real_day_level/r3_gex_regressions.csv`. Shown with controls for regime (each regime has its own level and its own expiry effect), because the participant-based GEX drifts upward over the sample (figure `results/pipeline_nse/figures/gex_by_sign_rule.png`).

| Sign rule | GEX on ordinary days (coefficient, p) | Extra GEX effect on expiry days (coefficient, p) | Total effect on expiry days |
| --- | --- | --- | --- |
| US convention | -0.38 (p = 0.030) | +0.39 (p = 0.11) | about 0 |
| Pro participant | -2.07 (p = 0.033) | +4.93 (p = 0.002) | +2.86 |
| Mirror of Client | -2.35 (p = 0.013) | +3.65 (p = 0.002) | +1.30 |

In plain words: on ordinary days, when dealers (by any of the three guesses) held more positive gamma the evening before, the next day was calmer, which is what hedging theory predicts (dealers who are long gamma sell into rallies and buy into dips). On expiry days the sign flips: more positive gamma before expiry goes with a more volatile expiry day. One possible reading is that the gamma of the expiring contract is concentrated near the money and disappears during the day, so hedges are unwound in size. We have not tested that mechanism.

How much to trust this: (1) the sign of dealer positions is not observed, only guessed; the Pro and Client weights come from all index options, not Nifty strikes; (2) it is in-sample, and R1 shows the same information does not improve out-of-sample forecasts; (3) we tried 3 sign rules with and without controls; (4) positioning may respond to expected volatility (reverse causality). With the regime controls, adding GEX raises $R^2$ only from 0.467 to between 0.469 and 0.474.

---

## R4. Pinning, first look

On expiry days, does the official close move toward the strike with the largest open interest (or the max-pain strike, both from the previous evening's chain) more than on other days? File: `results/real_day_level/r4_pinning.csv`. No: the change in distance to the max-OI strike does not differ between expiry days and other days (Mann-Whitney p = 0.67), and for max pain p = 0.12. This is a crude check. The dedicated study (`docs/PINNING_STUDY.md`) measures distance to the nearest strike instead and finds pinning in some periods only: Nifty closed 4.5 points closer to a strike on expiry days from November 2024 to August 2025 (p = 0.0006), not at all since the move to Tuesday, and Bank Nifty pinned while it had weekly options (p = 0.008). None of the dealer-gamma proxies explains when pinning happens.

---

## R5. End-of-day momentum and dealer gamma

**The question.** In US index futures, the return over most of the day tends to continue in the last 30 minutes ("intraday momentum", Gao, Han, Li and Zhou 2018), and Baltussen, Da, Lammers and Martens (2021) argue that part of the reason is option dealers who are short gamma: they have to buy after a rise and sell after a fall, and much of that trading happens near the close. If that holds in India, the effect should be stronger when dealers were short gamma the evening before.

**Setup.** For each day, the "rest of day" return runs from the previous day's last 1-minute close to the 14:59 bar close (so it includes the overnight gap), and the "last 30 minutes" return runs from 14:59 to 15:29. Both are from the validated Nifty 1-minute bars and are in basis points. We regress the last-30-minute return on the rest-of-day return, its product with an indicator for "GEX was negative at the previous close", and that indicator. Newey-West standard errors with 5 lags. 932 days, 198 of them expiry days. Files: `results/momentum_gex/regressions.csv`, `gamma_states.csv`, `stability.csv` and `results/momentum_gex_stdout.txt`.

**Result 1: no momentum, a mild reversal.** On all days the slope is -0.029 (p = 0.015): a 100 bp rise over the day is followed, on average, by a 2.9 bp fall in the last 30 minutes. The slope is about the same on expiry days (-0.037, p = 0.019). So Nifty does not show the US pattern at all over 2023 to 2026.

**Result 2: the reversal is concentrated on negative-GEX days, under two of the three sign rules.** The table shows the extra slope on days after negative GEX (the hedging story predicts a positive number).

| Sign rule | Days with GEX < 0 | Extra slope when GEX < 0 (p) | Same, with regime-specific slopes (p) | Same, with VIX and move-size controls (p) |
| --- | --- | --- | --- | --- |
| US convention | 107 | +0.013 (0.59) | +0.002 (0.95) | -0.001 (0.97) |
| Pro participant | 463 | -0.068 (0.0002) | -0.069 (0.0006) | -0.050 (0.003) |
| Mirror of Client | 38 | -0.072 (0.00003) | -0.069 (0.001) | +0.003 (0.95) |

In plain words: under the Pro rule, after a negative-GEX evening, a 100 bp move over the day is followed by about a 5.8 bp move the other way in the last 30 minutes; after a positive-GEX evening there is essentially no follow-through either way (slope +0.011, p = 0.29). The Pro result has the same sign in all three expiry regimes and in both halves of the sample (`stability.csv`; p between 0.0005 and 0.17 in the regimes, 0.03 and 0.01 in the halves). The Client-mirror result disappears once VIX and the size of the day's move are controlled for: it rests on only 38 days, most of them volatile ones, so it is better read as "big, volatile days reverse more" than as a gamma effect.

**Why this is puzzling.** Hedging theory says dealers who are long gamma trade against the market (which produces reversal) and dealers who are short gamma trade with it (momentum). If the Pro-based sign were right, negative GEX should mean momentum, but we see the strongest reversal there. If instead the Pro sign were backwards, R5 would fit the theory, but then R3's ordinary-day result (higher Pro GEX, calmer next day) would go against it. Both cannot hold under one simple story. Three readings we have not yet separated: (a) the Pro category's net option position is not a good proxy for who hedges Nifty options (it pools all index options and includes proprietary traders who may not delta-hedge at all); (b) negative Pro GEX picks out market conditions (for example, heavy retail call buying) in which the close reverses for reasons that have nothing to do with hedging; (c) chance, since we tried three rules and several specifications. The falsification tests planned next (placebo GEX dates, shuffled GEX, an out-of-sample check on later data) are designed to tell these apart.

**Closing auction.** From 3 August 2026 the 15:29 bar is the auction price after 13 frozen minutes, which changes what the last-30-minute return means. Without those 63 days the results are the same: overall slope -0.030 (p = 0.017), Pro interaction -0.072 (p = 0.0001) and -0.053 (p = 0.001) with controls (`results/closing_auction/r5_without_auction.csv`).

**Caveats.** Before costs, not a trading result: the average "follow the day's direction" payoff on negative-Pro-GEX days is -2.1 bp per day (t = -2.3), smaller than a typical round-trip cost in Nifty futures plus slippage. The 15:29 bar close is the last trade, not NSE's official close, which is a 30-minute average.
