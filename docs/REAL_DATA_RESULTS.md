# Real-data results

**Everything in this document comes from REAL market data:** NSE's own files (F&O bhavcopy, participant-wise open interest, daily index values including India VIX), downloaded from NSE's archive server, plus third-party 1-minute Nifty 50 bars from a public GitHub dataset that we checked day by day against NSE's official values. Simulation results are in `docs/EXPERIMENT_REPORT.md` (E1 to E7) and are kept separate.

Sample: 942 trading days, 2 January 2023 to 2 November 2026, of which 200 are Nifty expiry days. Out-of-sample forecasts start after a 150-day training period, so 792 test days. Every number below is copied from a file in `results/`; the file is named next to each table.

| ID | Question | Command | Output |
| --- | --- | --- | --- |
| R0 | Are the data right? | `python scripts/validate_minute_bars.py`, `python scripts/check_oi_units.py` | `results/data_validation/` |
| R1 | Out of sample, which model forecasts Nifty's intraday volatility best, and do dealer-gamma (GEX) features help? | `python scripts/run_pipeline.py --source nse` | `results/pipeline_nse/` |
| R2 | How much extra volatility does an expiry day carry, and did that change under the later expiry rules? | `python experiments/real_day_level.py` | `results/real_day_level/` |
| R3 | In-sample, is yesterday's GEX related to today's volatility? | same | same |
| R4 | A first, crude look at pinning | same | same |

More real-data studies are being added: a replication of the paper's Tables 1 to 6 on its own window, a proper pinning study, an event study of four expiry rule changes, and a stronger version of R1. Each gets its own document and is linked here when it lands.

---

## Findings (short)

1. **The data hold up.** Minute-bar highs and lows match NSE's official values within 0.05% on 943 of 943 days (highs) and 942 of 943 (lows) for Nifty. Bhavcopy open interest is in shares and adds up exactly to NSE's participant totals once contracts expiring that day are left out.
2. **The paper's model can be beaten out of sample.** A hybrid that keeps mcsGARCH (the paper's model) and corrects it with a LightGBM model of its errors forecasts the next 15 and 30 minutes about 7% and 9% better (QLIKE loss), on all days and on expiry days, with Diebold-Mariano p-values below 0.001. mcsGARCH is excluded from the Model Confidence Set for these horizons.
3. **For the last hour of the day, nothing beats a simple HAR model.** HAR is about 11% better than mcsGARCH there (p = 0.015), and the machine-learning models add nothing.
4. **GEX features do not improve out-of-sample forecasts** under any of the three sign rules. This matches what our simulations (E4) said to expect when the sign of dealer positions is uncertain and the effect is small.
5. **But GEX is related to volatility in-sample, with a sign that flips on expiry day.** On ordinary days, higher dealer gamma the evening before goes with lower volatility the next day, as hedging theory predicts. On expiry days the relation reverses. This holds under all three sign rules and after controlling for the expiry regime, but it is an in-sample association on a proxied sign, not a causal result (details in R3).
6. **Expiry days are more volatile than the rest of their week, but not in every regime.** About 11% more realized variance before November 2024, no clear difference from November 2024 to August 2025 (Thursday expiry, one weekly per exchange), and about 21% more since Nifty moved to Tuesday, concentrated in the last hour (about 43% more).

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
| LightGBM, with GEX | -4.6 | -7.6 | -4.8 | -7.4 | +1.8 | -8.7 |
| Hybrid, no GEX | **-6.7** | -6.2 | -9.0 | -6.6 | -1.3 | +3.2 |
| Hybrid, GEX (US sign) | -6.7 | -6.3 | **-9.3** | -5.6 | -1.7 | +2.7 |
| Hybrid, GEX (Pro participant sign) | -6.6 | -6.5 | -9.2 | -6.0 | -1.3 | +2.4 |

Test anchors: 19,800 (next 15), 19,008 (next 30), 792 (last hour); expiry days are 169 of the 792 test days.

**Is the improvement real?** Diebold-Mariano tests on daily average losses (`tests_*.csv`):

* Hybrid (no GEX) vs mcsGARCH: p < 0.0001 at 15 and 30 minutes on all days, p = 0.0007 and 0.0002 on expiry days. Not significant for the last hour (p = 0.26).
* LightGBM (no GEX) vs mcsGARCH on expiry days: p = 0.001 (15 min) and 0.0001 (30 min).
* HAR vs mcsGARCH, last hour: p = 0.015 (all days), 0.07 (expiry days).
* The Model Confidence Set (90%) excludes mcsGARCH, HAR, VIX and the naive models at 15 and 30 minutes; for the last hour it keeps almost every model, so the data cannot separate them there.

**Does GEX help?** The clean test compares the same model with and without the GEX block. For the hybrid, the extra loss reduction from GEX is zero to within noise under every sign rule (p between 0.22 and 0.88). For LightGBM, GEX gives a small gain at 15 minutes (0.7 points, p = 0.044) that does not hold at 30 minutes (p = 0.064), in the last hour (worse), or in the hybrid. We read this as no reliable out-of-sample value from end-of-day GEX.

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

(Percentages are $e^{\text{mean}} - 1$ of the log differences.) The Tuesday period differs from the Thursday single-weekly period (Mann-Whitney p = 0.012 for the whole day). Each period bundles several changes (SEBI's November 2024 package also raised contract sizes and added margin on expiry day), so these are descriptive, and three regimes times three measures is nine tests. A fuller event study with Bank Nifty as a comparison index is in progress.

---

## R3. Is yesterday's GEX related to today's volatility? (in-sample)

Regression of the log of today's realized variance on yesterday's log realized variance, its 5-day average, the log of yesterday's VIX, an expiry-day dummy, yesterday's net GEX (divided by its 20-day median gross GEX, so it is a share between about -0.3 and +0.6) and GEX times the expiry dummy. Newey-West standard errors with 5 lags, 937 days. File: `results/real_day_level/r3_gex_regressions.csv`. Shown with controls for regime (each regime has its own level and its own expiry effect), because the participant-based GEX drifts upward over the sample (figure `results/pipeline_nse/figures/gex_by_sign_rule.png`).

| Sign rule | GEX on ordinary days (coefficient, p) | Extra GEX effect on expiry days (coefficient, p) | Total effect on expiry days |
| --- | --- | --- | --- |
| US convention | -0.38 (p = 0.030) | +0.39 (p = 0.12) | about 0 |
| Pro participant | -2.07 (p = 0.033) | +4.93 (p = 0.002) | +2.86 |
| Mirror of Client | -2.35 (p = 0.013) | +3.65 (p = 0.002) | +1.30 |

In plain words: on ordinary days, when dealers (by any of the three guesses) held more positive gamma the evening before, the next day was calmer, which is what hedging theory predicts (dealers who are long gamma sell into rallies and buy into dips). On expiry days the sign flips: more positive gamma before expiry goes with a more volatile expiry day. One possible reading is that the gamma of the expiring contract is concentrated near the money and disappears during the day, so hedges are unwound in size. We have not tested that mechanism.

How much to trust this: (1) the sign of dealer positions is not observed, only guessed; the Pro and Client weights come from all index options, not Nifty strikes; (2) it is in-sample, and R1 shows the same information does not improve out-of-sample forecasts; (3) we tried 3 sign rules with and without controls; (4) positioning may respond to expected volatility (reverse causality). With the regime controls, adding GEX raises $R^2$ only from 0.467 to between 0.469 and 0.473.

---

## R4. Pinning, first look

On expiry days, does the official close move toward the strike with the largest open interest (or the max-pain strike, both from the previous evening's chain) more than on other days? File: `results/real_day_level/r4_pinning.csv`. No: the change in distance to the max-OI strike does not differ between expiry days and other days (Mann-Whitney p = 0.67), and for max pain p = 0.12. This is a crude check; a dedicated pinning study is in progress.
