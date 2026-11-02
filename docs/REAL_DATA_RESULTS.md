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
| R1b | Where do the forecasting gains come from (time of day, regime, tails, MSE)? | `python experiments/forecast_diagnostics.py` | `results/forecast_diagnostics/` |

Separate real-data studies, each with its own document:

* `docs/REPLICATION_NSE.md`: the paper's Tables 1 to 6 rebuilt on NSE data. The "after" sample matches closely; the stated "before" window does not, and a 64-day window from 14 November 2022 to 10 February 2023 reproduces the printed "before" numbers (our main question for the authors).
* `docs/PINNING_STUDY.md`: a full pinning study (replaces the crude R4 check below).
* `docs/INDEPENDENT_REVIEW.md`: an independent referee-style review of this document and the two above. Its points are addressed here (R1 diagnostics, R3 scale-free versions, multiple-testing columns) or handed to the studies still running.
* `docs/EVENT_STUDY.md`: five expiry rule changes (2019 to 2026) with Nifty and Bank Nifty as each other's comparison and placebo dates. The whole-day expiry premium barely moves with the rules; the end of the day does (Bank Nifty's move to Wednesday raised its last-30-minute premium by 1.14 log points, placebo p 0.005). The per-minute Rogers-Satchell measure the paper uses shows an expiry premium that 5-minute realized variance does not.
* `docs/FALSIFICATION.md`: seven families of tests that try to break R3 and R5 (shuffles, timing placebos, random sign rules, out of sample, Bank Nifty, a quasi-causal roll-off test, multiple testing). R5 survives all of them; R3's ordinary-day term does not.
* `docs/DEALER_SIGN.md`: the dealer sign estimated from the market's response instead of assumed. A documented negative result: the two outcomes point to nearly opposite signs.
* `docs/INTRADAY_GAMMA.md`: expiry-day gamma re-priced minute by minute at the current index level. A clean null on expiry days, also in the final half hour before the close (pre-registered follow-up: 0.26, p = 0.39); the one lead that passed its placebos (gamma concentrated near the index) did not replicate on Bank Nifty (p = 0.66).
* Still running: a stronger version of R1 (option-implied features, more frequent mcsGARCH refits).
* `docs/CONNECTEDNESS_GAMMA.md`: does Nifty and Bank Nifty volatility spill over more on short-gamma days (a new question for the paper's connectedness design)? The pre-stated Pro-rule test is borderline (+4.29 points on the paper's scale, permutation p = 0.044) and does not survive a VIX control or most robustness checks; total gamma (no sign) goes with lower connectedness (Holm p = 0.010), like the level effect in FALSIFICATION.md.
* `docs/SENSEX_CONFOUND.md`: is BSE's Sensex expiry a hidden confounder? BSE's servers are blocked, so the Sensex calendar is reported, not verified (it matches all 71 option-price expiry dates available). The pre-stated test is not significant (p = 0.16), but Nifty's last 30 minutes are more volatile on Sensex expiry days. With Sensex days left out of the weekly baseline, R2's premia get a little larger (Tuesday last hour +43% becomes +63%), so the published NSE effects were understated, not overstated.
* `docs/F2_ECONOMICS.md`: is R5 worth anything after trading costs? A pre-registered walk-forward rule (design committed before any data were used) on 90 test trades from 2025 to July 2026 earns +4.15 bp per trade before costs but -0.89 bp after a base-case 5 bp round trip; Deflated Sharpe Ratio far below the 0.95 bar. Verdict by the pre-stated rule: not economically meaningful. R5 is a statistical regularity, not a trading opportunity.
* `docs/PINNING_MECHANISM.md`: which market-structure variable explains when pinning appears? Pre-registered (design committed before any outcome data were used). The primary candidate, open interest near the money, has the wrong sign (p = 0.010), so it is ruled out; no candidate explains the regime pattern.
* `docs/TEST_LEDGER.md` (built by `scripts/build_test_ledger.py`): every one of the project's 4,930 hypothesis tests, whether it was pre-specified, its raw p-value and its Benjamini-Yekutieli p-value across the whole project (valid under any dependence between tests). Of the findings below, finding 2 survives this correction in full; findings 6, 7 (the in-sample R5 test) and 8 (outcome b) survive in part; finding 5 does not.
* `docs/POWER_REAL.md`: how large an effect each test could have detected, by simulation on the real data. R5 and R3 had about 80% power only for effects close to the ones estimated; effects half as large would need 5 to 11 more years of data. A GEX sign wrong on a quarter of days cuts power sharply.
* `docs/REKEY_CHECK.md`: a second person re-typed all 223 printed numbers from the paper blind; 222 matched, and the one difference is a typesetting slip in the paper.

---

## Findings (short)

1. **The data hold up.** Minute-bar highs and lows match NSE's official values within 0.05% on 943 of 943 days (highs) and 942 of 943 (lows) for Nifty. Bhavcopy open interest is in shares and adds up exactly to NSE's participant totals once contracts expiring that day are left out.
2. **The paper's model can be beaten out of sample, mostly but not only at the open.** A hybrid that keeps mcsGARCH (the paper's model) and corrects it with a LightGBM model of its errors forecasts the next 15 and 30 minutes about 7% and 9% better on all days and about 6% better on expiry days (QLIKE loss), with Diebold-Mariano p-values of 0.001 or below. The gain is largest at 09:15 (16% and 27%), where mcsGARCH leans on yesterday's daily variance, and is still about 6% and 7% at all later times of day, in each of the three expiry regimes, and with the worst 1% of forecasts removed. mcsGARCH is excluded from the Model Confidence Set for these horizons. Under squared-error loss (MSE) the ranking reverses, because mcsGARCH and the hybrid made a few very large errors right after the 4 June 2024 election-result crash (R1).
3. **For the last hour of the day, no model reliably beats mcsGARCH.** Over all test days HAR looks about 11% better (one-sided p = 0.015), but most of that edge comes from the last 63 days, after NSE introduced a closing auction on 3 August 2026 (see R0). Before that date HAR is about 4% better and the difference is not significant (p = 0.11). The machine-learning models add nothing for the last hour.
4. **GEX features do not improve out-of-sample forecasts** under any of the three sign rules. This matches what our simulations (E4) said to expect when the sign of dealer positions is uncertain and the effect is small.
5. **In-sample, Pro-based GEX is related to volatility with a sign that flips on expiry day, but the falsification tests weaken this a lot.** On ordinary days, higher dealer gamma the evening before goes with lower volatility the next day, as hedging theory predicts; on expiry days the relation reverses. Under the US and Client-mirror rules it does not survive scale-free versions of GEX, so the earlier statement that it holds under all three rules was too strong. We then tried hard to break it ([FALSIFICATION.md](FALSIFICATION.md)). The ordinary-day part failed: a 20-day block shuffle of GEX gives p = 0.25, GEX measured on later days predicts as well as yesterday's, 91.9% of randomly drawn sign rules are also significant, and gross gamma alone (no sign at all) gives t = -5.3. So it behaves like a slow-moving level of total gamma, not like a dealer signal. The expiry-day flip under the Pro rule survived the shuffles, the random sign rules and a timing horse race (Holm p = 0.020 among the 18 headline tests), but adding it makes out-of-sample forecasts worse (squared error +5%), it does not replicate on Bank Nifty (p = 0.25), and it is carried by the daily Pro positioning weights from the participant file rather than by the option chain. It is an in-sample association on a proxied sign, not a causal or forecasting result (R3).
6. **Expiry days are more volatile than the rest of their week, but not in every regime.** About 11% more realized variance before November 2024 (p = 0.045, but the test over-rejects in that period, so this is weak), no clear difference from November 2024 to August 2025 (Thursday expiry, one weekly per exchange), and about 21% more since Nifty moved to Tuesday, concentrated in the last hour (about 43% more). BSE's Sensex expiry days were raising the weekly baseline; leaving them out makes every premium a little larger (Tuesday last hour about 63%; [SENSEX_CONFOUND.md](SENSEX_CONFOUND.md), calendar reported, not verified).
7. **Nifty reverses into the close rather than continuing, and the reversal is stronger on days when the Pro-based GEX is negative.** This is the opposite of what the US literature on hedging-driven momentum predicts. It is the most robust GEX result in the project: it survived all seven falsification tests (shuffles p = 0.0005 to 0.010, GEX from the next day does not predict it, Holm p = 0.016 among 78 tests, better out-of-sample forecasts on 2025 to 2026 data with a test fixed from 2023 to 2024 data, p = 0.006), and it is the same without the closing-auction days (R5). It is too small to trade: after realistic costs a pre-registered rule earns nothing ([F2_ECONOMICS.md](F2_ECONOMICS.md)).
8. **The data do not reveal one dealer sign that fits both predictions of hedging theory.** When the dealer sign is estimated from the market's response instead of assumed ([DEALER_SIGN.md](DEALER_SIGN.md)), the volatility outcome points weakly to the Pro sign (p = 0.18) and the end-of-day outcome points strongly to the reverse of the Pro sign (p = 4e-6). Out of sample, no estimated sign beats the three fixed rules after correcting for multiple tests. In other words, finding 7 fits hedging theory only if the Pro rule has the wrong sign, and finding 5 fits only if it has the right one; the honest summary is that the participant-based sign is a proxy whose meaning we cannot pin down with public data.

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

Setup: forecasts of realized variance (sum of squared 1-minute returns) over the next 15 minutes and next 30 minutes from every 15-minute mark, and over the last hour from 14:30. HAR, LightGBM and the hybrid's correction are refitted every 21 trading days and mcsGARCH (with the daily EGARCH that feeds it) every 63 trading days, each time on all data before the forecast period (expanding window), so no model ever sees the future. (The slower mcsGARCH refit mildly handicaps the paper's model; a like-for-like rerun with 21-day refits is part of the model study in progress.) Loss: QLIKE (lower is better; it punishes forecasts that are too low more than forecasts that are too high, and is the standard loss for variance forecasts), shown as % difference from mcsGARCH. Files: `results/pipeline_nse/scoreboard_*.csv`, `tests_*.csv`, `mcs_*.csv`.

| Model | Next 15 min, all days | Next 15 min, expiry days | Next 30 min, all days | Next 30 min, expiry days | Last hour, all days | Last hour, expiry days |
| --- | --- | --- | --- | --- | --- | --- |
| mcsGARCH (the paper's model) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| HAR | +5.2 | +4.0 | +6.7 | +5.0 | -10.8 | -10.2 |
| India VIX only | +52.5 | +33.8 | +63.6 | +42.2 | +17.6 | +27.7 |
| LightGBM, no GEX | -3.9 | -6.2 | -4.2 | -6.6 | -0.5 | -8.2 |
| LightGBM, with GEX | -4.4 | -7.2 | -4.7 | -7.1 | +1.9 | -8.8 |
| Hybrid, no GEX | **-6.7** | **-6.2** | **-9.0** | **-6.6** | -1.3 | +3.2 |
| Hybrid, GEX (US sign) | -6.7 | -6.3 | -9.2 | -5.8 | -1.6 | +3.4 |
| Hybrid, GEX (Pro participant sign) | -6.8 | -6.5 | -9.2 | -6.1 | -1.3 | +2.5 |

Test anchors: 19,800 (next 15), 19,008 (next 30), 792 (last hour); expiry days are 169 of the 792 test days.

**Is the improvement real?** Diebold-Mariano tests on daily average losses (`tests_*.csv`; one-sided, asking whether the named model has lower loss than mcsGARCH, with the Harvey, Leybourne and Newbold small-sample correction):

* Hybrid (no GEX) vs mcsGARCH: p < 0.0001 at 15 and 30 minutes on all days, p = 0.0007 and 0.0002 on expiry days. Not significant for the last hour (p = 0.26).
* LightGBM (no GEX) vs mcsGARCH on expiry days: p = 0.001 (15 min) and 0.0001 (30 min).
* HAR vs mcsGARCH, last hour: p = 0.015 (all days), 0.07 (expiry days). Without the 63 closing-auction days: HAR is 4.2% better on all days (p = 0.11) and 7.1% on expiry days (p = 0.17), so the last-hour edge is not established (`results/closing_auction/r1_dm_tests.csv`).
* Without the closing-auction days the hybrid's gain at 15 and 30 minutes is about the same: 6.0% and 7.8% on all days, 5.5% and 5.7% on expiry days, all with p of 0.0013 or smaller.
* The Model Confidence Set (90%) excludes mcsGARCH, HAR, VIX and the naive models at 15 and 30 minutes; for the last hour it keeps almost every model, so the data cannot separate them there.
* For the last hour, HAR against the hybrid (two-sided): p = 0.019 on all days, 0.062 before the auction, 0.065 on expiry days (`results/forecast_diagnostics/tails_and_last_hour_tests.csv`).

**Where the gain comes from** (`python experiments/forecast_diagnostics.py`, files in `results/forecast_diagnostics/`). QLIKE, % against mcsGARCH (negative = better):

| Subset | Hybrid, next 15 min | Hybrid, next 30 min | LightGBM, next 15 min | LightGBM, next 30 min |
| --- | --- | --- | --- | --- |
| All anchors (the headline) | -6.7 | -9.0 | -3.9 | -4.2 |
| 09:15 anchor only | -16.4 | -26.9 | -18.4 | -25.7 |
| All anchors after 09:15 | -5.8 | -7.1 | -2.5 | -1.9 |
| Worst 1% of mcsGARCH's losses removed | -6.2 | -7.9 | -2.1 | -1.9 |
| Several weeklies (to Nov 2024) | -5.4 | -6.9 | +2.8 | +4.8 |
| One weekly, Thursday | -7.0 | -8.6 | -6.2 | -8.3 |
| Tuesday expiry | -7.7 | -11.8 | -8.5 | -10.8 |

So the hybrid's advantage is largest at the open, where mcsGARCH's forecast leans on yesterday's daily variance, but it holds at later times of day and in every regime. LightGBM on its own is not better than mcsGARCH in the first regime. The hybrid has the lower loss on 52.5% (15 minutes) and 53.2% (30 minutes) of individual anchors: a steady small edge, not a few big wins.

**MSE tells a different story, for a specific reason.** Under mean squared error, mcsGARCH (2.3e-10 at 15 minutes) and the hybrid (1.5e-10) are far worse than HAR, LightGBM and VIX (all about 3.5e-11). Ten anchors account for 97% of mcsGARCH's squared error at 15 minutes, led by 5 June 2024 at 09:15, the morning after the election-result crash: mcsGARCH forecast a 15-minute variance of 0.00201 against a realized 0.000099, about 20 times too high, and the hybrid inherited most of it (0.00157), while LightGBM did not (0.000008) (`largest_mcsgarch_errors.csv`). Our reading, not yet checked component by component, is that after the crash both the daily component and the intraday GARCH component of mcsGARCH were high, and the model multiplies them. With the 09:15 anchor removed, the MSE gap shrinks (mcsGARCH 1.5e-11, hybrid 1.2e-11, HAR 9.4e-12). QLIKE is the better loss for comparing variance forecasts (Patton 2011), but a risk manager would care about that one morning, and a cap on the daily factor after extreme days is an obvious fix to test.

**Does GEX help?** The clean test compares the same model with and without the GEX block. For the hybrid, the extra loss reduction from GEX is zero to within noise under every sign rule (one-sided Diebold-Mariano p between 0.11 and 0.87 across horizons and day types). For LightGBM, GEX is about half a point better at 15 and 30 minutes, which is not significant (p = 0.13 and 0.09) and reverses in the last hour. We read this as no reliable out-of-sample value from end-of-day GEX.

**What the mcsGARCH fit looks like on the whole sample** (`mcsgarch_full_sample.csv`): intraday GARCH $\alpha = 0.049$, $\beta = 0.945$, so persistence 0.994 and an implied half-life of about 2 hours; Student-$t$ degrees of freedom 5.9. Across the 13 walk-forward refits $\alpha$ ranges from 0.025 to 0.051 (`mcsgarch_params.csv`).

**Caveats.** Minute volume does not exist for the index before 2026, so the volume feature is constant and the "order-flow" feature `ofi15` reduces to a price-based measure (the signed size of recent returns), not true order flow; its scale constant is the median squared return over the whole sample, a single number but technically look-ahead (being fixed in the model study). The GEX features are built from end-of-day chains only. The regime label is a LightGBM feature, so in the first block after each rule change the model has not seen that label. The risk-free rate is fixed at 6.5%; gamma barely depends on it and forwards come from put-call parity. The Diebold-Mariano p-values use the normal distribution after the small-sample correction (Student t would change them only in the third decimal for 169 expiry days). Over 84 Diebold-Mariano tests are behind the R1 tables; the Model Confidence Set is the multiple-testing-safe summary.

---

## R2. Expiry-day volatility by expiry regime

For each week we take the expiry day's log realized variance minus the average log realized variance of the other days of that week (this cancels market-wide shocks that last the whole week). File: `results/real_day_level/r2_expiry_effect_by_regime.csv`.

| Regime | Weeks | Whole day: expiry vs rest of week | 95% interval | p (signed-rank) | Last hour: expiry vs rest of week | p |
| --- | --- | --- | --- | --- | --- | --- |
| Several weeklies, Thursday (to 19 Nov 2024) | 98 | +11% | +1% to +22% | 0.045 | +16% | 0.053 |
| One weekly per exchange, Thursday (20 Nov 2024 to 31 Aug 2025) | 41 | -11% | -26% to +8% | 0.26 | +10% | 0.65 |
| Nifty expiry on Tuesday (from 1 Sep 2025) | 61 | +21% | +8% to +38% | 0.005 | +43% | 0.00005 |

(Percentages are $e^{\text{mean}} - 1$ of the log differences.) **Size check** ([POWER_REAL.md](POWER_REAL.md), simulations on the real data with no effect added): the signed-rank test holds its 5% level in the Tuesday period, but over-rejects in the two Thursday periods (8% to 9% false positives for several weeklies, 13% to 18% for one weekly, because weekly premiums there are autocorrelated). So the several-weeklies p = 0.045 is weaker evidence than it looks; the Tuesday result is not affected. The Tuesday result does not come from the closing auction: on the 48 Tuesday expiry weeks before 3 August 2026 the whole-day effect is +24% (p = 0.005) and the last-hour effect +42% (p = 0.00004) (`results/closing_auction/r2_tuesday_split.csv`). The Tuesday period differs from the Thursday single-weekly period (Mann-Whitney p = 0.012 for the whole day). Each period bundles several changes (SEBI's November 2024 package also raised contract sizes and added margin on expiry day), so these are descriptive. The file holds 18 tests (three measures in three regimes, plus three pairwise regime comparisons per measure); the Thursday multi-weekly whole-day result (p = 0.045) would not survive a correction for that many, while the Tuesday last-hour result (p = 0.00005) would. The event study (`docs/EVENT_STUDY.md`) looks at each rule change with placebo dates and Bank Nifty as a comparison.

---

## R3. Is yesterday's GEX related to today's volatility? (in-sample)

Regression of the log of today's realized variance on yesterday's log realized variance, its 5-day average, the log of yesterday's VIX, an expiry-day dummy, yesterday's net GEX (divided by its 20-day median gross GEX; this ratio runs from -0.31 to +0.62 under the US rule but only from -0.10 to +0.10 under the Pro rule and from -0.09 to +0.19 under the Client mirror, so coefficients are not comparable across rules) and GEX times the expiry dummy. Newey-West standard errors with 5 lags, 937 days. File: `results/real_day_level/r3_gex_regressions.csv`. Shown with controls for regime (each regime has its own level and its own expiry effect), because the participant-based GEX drifts upward over the sample (figure `results/pipeline_nse/figures/gex_by_sign_rule.png`).

| Sign rule | GEX on ordinary days (coefficient, p) | Extra GEX effect on expiry days (coefficient, p) | Total effect on expiry days |
| --- | --- | --- | --- |
| US convention | -0.38 (p = 0.030) | +0.39 (p = 0.11) | about 0 |
| Pro participant | -2.07 (p = 0.033) | +4.93 (p = 0.002) | +2.86 |
| Mirror of Client | -2.35 (p = 0.013) | +3.65 (p = 0.002) | +1.30 |

In plain words: on ordinary days, when dealers (by any of the three guesses) held more positive gamma the evening before, the next day was calmer, which is what hedging theory predicts (dealers who are long gamma sell into rallies and buy into dips). On expiry days the sign flips: more positive gamma before expiry goes with a more volatile expiry day. One possible reading is that the gamma of the expiring contract is concentrated near the money and disappears during the day, so hedges are unwound in size. We have not tested that mechanism.

**Robustness to how GEX is scaled** (`results/real_day_level/r3_robustness.csv`). The ratio above has a problem: gross gamma peaks the evening before expiry (on average 1.1e12 rupees per 1% move with one trading day to expiry against 3.2e11 with five), and a 20-day median does not remove that weekly swing, so the regressor is mechanically about four times larger on expiry eves, which could feed the expiry interaction. Five variants remove that channel; all keep the regime controls. Holm-adjusted p-values are over all 30 GEX terms in the table.

| Variant | US convention: ordinary / extra on expiry (p) | Pro participant: ordinary / extra on expiry (p, Holm p) | Client mirror: ordinary / extra on expiry (p) |
| --- | --- | --- | --- |
| Per standard deviation | -0.040 (0.03) / +0.041 (0.11) | -0.053 (0.03) / +0.126 (0.002, Holm 0.04) | -0.071 (0.01) / +0.110 (0.002) |
| Sign only (GEX > 0) | +0.017 (0.73) / -0.024 (0.82) | -0.088 (0.04) / +0.332 (0.002, Holm 0.06) | -0.098 (0.50) / +0.131 (0.44) |
| Net / gross of the same chain | +0.27 (0.35) / +0.23 (0.62) | -3.16 (0.004) / +8.21 (0.001, Holm 0.04) | -0.50 (0.71) / +3.11 (0.09) |
| Same, plus gross gamma controls | +0.31 (0.27) / +0.23 (0.61) | -3.06 (0.004) / +8.14 (0.001, Holm 0.04) | -1.49 (0.25) / +3.86 (0.03) |
| Same, before the closing auction | +0.37 (0.23) / +0.32 (0.52) | -3.51 (0.001) / +8.94 (0.0005, Holm 0.02) | -0.69 (0.61) / +3.27 (0.08) |

So the sign flip is robust under the Pro rule: with a scale-free measure, with gross gamma controlled, and without the auction days, and its expiry term survives the Holm correction in four of five variants (the sign-only version gives Holm p = 0.06). Under the US rule it disappears once GEX is scale-free, and under the Client mirror it weakens to borderline. In the Pro sign-only version, an evening with positive Pro GEX goes with about 8% lower variance the next ordinary day and about 28% higher variance the next expiry day ($e^{-0.088} - 1$ and $e^{0.244} - 1$).

How much to trust this: (1) the sign of dealer positions is not observed, only guessed; the Pro and Client weights come from all index options, not Nifty strikes; (2) it is in-sample, and R1 shows the same information does not improve out-of-sample forecasts; (3) we tried 3 sign rules, with and without controls, in six versions of the GEX measure (the Holm column above accounts for the robustness table, not for every version we looked at); (4) positioning may respond to expected volatility (reverse causality). With the regime controls, adding GEX raises $R^2$ only from 0.467 to between 0.469 and 0.474.

**After the falsification tests** ([FALSIFICATION.md](FALSIFICATION.md), REAL DATA). The ordinary-day column does not survive: a 20-day block shuffle of the GEX series gives p = 0.25, GEX from later days predicts today's volatility at least as well as yesterday's, 91.9% of randomly drawn sign rules give a significant coefficient, and gross gamma alone gives t = -5.3. It is a slow-moving level effect (more total option gamma, calmer markets), not evidence about which way dealers hedge. The expiry-day flip under the Pro rule survives the shuffles, the random sign rules and the timing horse race, but it is carried by the daily Pro positioning weights, makes out-of-sample forecasts about 5% worse, and does not replicate on Bank Nifty (p = 0.25). Treat it as a lead to test on new data, not as a result.

**How much power did R3 have?** ([POWER_REAL.md](POWER_REAL.md), SIMULATIONS CALIBRATED ON REAL DATA.) With 937 days, the Pro-share effects these regressions detect 80% of the time are about 3.0 (ordinary days) and 6.9 to 7.1 (expiry interaction), just below the estimates (-3.16 and 8.21). Effects half as large would need about 3.1 to 3.9 times the sample (8.0 to 10.9 more years), and a GEX sign wrong on 25% of random days cuts power at the estimate to 23% and 27%.

---

## R4. Pinning, first look

On expiry days, does the official close move toward the strike with the largest open interest (or the max-pain strike, both from the previous evening's chain) more than on other days? File: `results/real_day_level/r4_pinning.csv`. No: the change in distance to the max-OI strike does not differ between expiry days and other days (Mann-Whitney p = 0.67), and for max pain p = 0.12. This is a crude check. The dedicated study (`docs/PINNING_STUDY.md`) measures distance to the nearest strike instead and finds pinning in some periods only: Nifty closed 4.5 points closer to a strike on expiry days from November 2024 to August 2025 (p = 0.0006), not at all since the move to Tuesday, and Bank Nifty pinned while it had weekly options (p = 0.008). None of the dealer-gamma proxies explains when pinning happens.

---

## R5. End-of-day momentum and dealer gamma

**The question.** In US markets the return earlier in the day tends to continue into the last 30 minutes ("intraday momentum": Gao, Han, Li and Zhou 2018 found it for the S&P 500 ETF using the first half hour; Baltussen, Da, Lammers and Martens 2021 used the rest-of-day return in futures markets, the design we copy), and Baltussen et al. argue that part of the reason is option dealers who are short gamma: they have to buy after a rise and sell after a fall, and much of that trading happens near the close. If that holds in India, the effect should be stronger when dealers were short gamma the evening before.

**Setup.** For each day, the "rest of day" return runs from the previous day's last 1-minute close to the 14:59 bar close (so it includes the overnight gap), and the "last 30 minutes" return runs from 14:59 to 15:29. Both are from the validated Nifty 1-minute bars and are in basis points. We regress the last-30-minute return on the rest-of-day return, its product with an indicator for "GEX was negative at the previous close", and that indicator. Newey-West standard errors with 5 lags. 932 days, 198 of them expiry days; the GEX regressions use the 929 of them (197 expiry) on which GEX at the previous close exists. Files: `results/momentum_gex/regressions.csv`, `gamma_states.csv`, `stability.csv` and `results/momentum_gex_stdout.txt`.

**Result 1: no momentum, a mild reversal.** On all days the slope is -0.029 (p = 0.015): a 100 bp rise over the day is followed, on average, by a 2.9 bp fall in the last 30 minutes. The slope is about the same on expiry days (-0.037, p = 0.019). So Nifty does not show the US pattern at all over 2023 to 2026.

**Result 2: the reversal is concentrated on negative-GEX days, under two of the three sign rules.** The table shows the extra slope on days after negative GEX (the hedging story predicts a positive number).

| Sign rule | Days with GEX < 0 | Extra slope when GEX < 0 (p) | Same, with regime-specific slopes (p) | Same, with VIX and move-size controls (p) |
| --- | --- | --- | --- | --- |
| US convention | 107 | +0.013 (0.59) | +0.002 (0.95) | -0.001 (0.97) |
| Pro participant | 463 | -0.068 (0.0002) | -0.069 (0.0006) | -0.050 (0.003) |
| Mirror of Client | 38 | -0.072 (0.00003) | -0.069 (0.001) | +0.003 (0.95) |

In plain words: under the Pro rule, after a negative-GEX evening, a 100 bp move over the day is followed by about a 5.7 bp move the other way in the last 30 minutes; after a positive-GEX evening there is essentially no follow-through either way (slope +0.011, p = 0.29). The Pro result has the same sign in all three expiry regimes and in both halves of the sample (`stability.csv`; p between 0.0005 and 0.17 in the regimes, 0.035 and 0.011 in the halves). Across all 36 GEX interaction tests in `regressions.csv` (three rules, four specifications, three day subsets), the Pro result keeps a Holm-adjusted p of 0.007 (plain), 0.021 (regime slopes) and 0.077 (VIX and size controls); Benjamini-Hochberg adjusted 0.003, 0.005 and 0.010. The Client-mirror result disappears once VIX and the size of the day's move are controlled for: it rests on only 38 days, most of them volatile ones, so it is better read as "big, volatile days reverse more" than as a gamma effect.

**Why this is puzzling.** Hedging theory says dealers who are long gamma trade against the market (which produces reversal) and dealers who are short gamma trade with it (momentum). If the Pro-based sign were right, negative GEX should mean momentum, but we see the strongest reversal there. If instead the Pro sign were backwards, R5 would fit the theory, but then R3's ordinary-day result (higher Pro GEX, calmer next day) would go against it. Both cannot hold under one simple story. Three readings we have not yet separated: (a) the Pro category's net option position is not a good proxy for who hedges Nifty options (it pools all index options and includes proprietary traders who may not delta-hedge at all); (b) negative Pro GEX picks out market conditions (for example, heavy retail call buying) in which the close reverses for reasons that have nothing to do with hedging; (c) chance, since we tried three rules and several specifications. The falsification tests ([FALSIFICATION.md](FALSIFICATION.md)) rule out (c) as far as data can: the Pro result survives shuffled GEX, placebo dates, random sign rules and a pre-registered out-of-sample test. They also weaken R3, whose ordinary-day term turned out to be a proxy for gross gamma. Estimating the sign from the data ([DEALER_SIGN.md](DEALER_SIGN.md)) puts the end-of-day dealers at the reverse of the Pro sign, which is reading (a) or (b), not a confirmation of the hedging story.

**Closing auction.** From 3 August 2026 the 15:29 bar is the auction price after 13 frozen minutes, which changes what the last-30-minute return means. Without those 63 days the results are the same: overall slope -0.030 (p = 0.017), Pro interaction -0.072 (p = 0.0001) and -0.053 (p = 0.001) with controls (`results/closing_auction/r5_without_auction.csv`).

**Caveats.** Before costs, not a trading result: the average "follow the day's direction" payoff on negative-Pro-GEX days is -2.1 bp per day (t = -2.3), smaller than a typical round-trip cost in Nifty futures plus slippage. Before 3 August 2026 the 15:29 bar close is the last trade, not NSE's official close (then a 30-minute average); from that date it is the auction price and equals the official close.
