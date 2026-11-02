# Does dealer gamma at the current index level move Nifty volatility, minute by minute?

**REAL DATA.** NSE F&O bhavcopies, NSE participant-wise open interest and NSE official index values (India VIX included), 2 January 2023 to 2 November 2026, plus third-party 1-minute Nifty 50 and Nifty Bank bars that the project checked day by day against NSE (`results/data_validation/`). Every number in this document comes from a file written by `experiments/intraday_gamma.py` (folder `results/intraday_gamma/`, printed log `results/intraday_gamma_stdout.txt`). The file behind each table is named under it.

Written for Shreyan Thanki, to discuss with Prof. Piyush Pandey. No finance or statistics background is assumed: Section 1 explains each idea once.

## Findings first

All results are REAL DATA. "Coefficient" means the elasticity of the next 15 minutes' realized variance with respect to gamma at spot: a coefficient of 0.5 means that 1% more gamma goes with 0.5% more variance.

1. **On expiry days, volatility does not respond to gamma at spot (the main result, a clean null).** Coefficient 0.142, 95% interval -0.128 to 0.413, p = 0.30 (Holm-adjusted p = 0.90). A typical within-day swing in gamma at spot (one standard deviation, 8.3%) goes with about 1.2% more variance; the interval rules out a rise of more than about 3.4% or a fall of more than about 1.1% per such swing. The null holds with a 60- or 120-minute time floor, with a clock that ends at 15:10, before the closing auction, in each of the three expiry regimes, and for the next 30 minutes.
2. **On the day before expiry and on ordinary days the pre-specified regression is significant, but it is not hedging.** Coefficients -0.723 (interval -1.124 to -0.322, p = 0.0005, Holm 0.003) and -0.579 (-1.044 to -0.113, p = 0.015, Holm 0.09). Two checks say the cause is something else. First, the half-strike placebo and another day's OI profile give the same coefficients (for example -0.80 for the median "another day" draw on the day before expiry, with 98% of 200 draws significant). Second, the coefficients vanish (0.266, p = 0.50 and -0.296, p = 0.46) once the regression also controls for how far and in which direction the index has moved since the previous close. Gross gamma at spot rises within the day when the index rises (correlation 0.63 and 0.72), partly because GEX is defined with the index level squared, and volatility falls after the index rises (the "leverage effect"). The regression was picking that up.
3. **The signed versions reveal no sign either.** On expiry days: US convention -0.130 (p = 0.11), NSE "Pro" -3.70 (p = 0.36), mirror of NSE "Client" +3.33 (p = 0.044, Holm 0.22). The large signed coefficients on other days (for example US convention -0.845, p < 0.001 on the day before expiry) are again the leverage effect: with the move since the previous close controlled, no signed coefficient on any day type has p below 0.06.
4. **Expiry afternoons (13:00 to 15:00) show a positive coefficient that does not survive checks.** 0.534 (0.046 to 1.022, p = 0.032), the direction expected if dealers were net short gamma. But the half-strike placebos give 0.392 and 0.601, another day's profile gives 0.336 (median of 200 draws), and with the leverage control the coefficient is 0.361 (p = 0.26). It is one of 117 tests.
5. **One lead worth a fresh test: how concentrated gamma is around spot.** On expiry days, when a larger share of gamma at spot comes from strikes within 0.5% of the index, the next 15 minutes are calmer: coefficient -0.386 (-0.727 to -0.044, p = 0.027; -0.450, p = 0.010 with the leverage control). Unlike the gamma level, this passes all three placebos (half-strike shift: -0.021, p = 0.91; another day's profile: randomization p = 0.010; permuted OI: randomization p = 0.025). The direction is the one expected if dealers are net long gamma near the money. It was a secondary measure, it does not survive a Holm correction within its family (adjusted p = 0.38 over the 19 secondary tests), the effect is small (about 2% of variance per within-day standard deviation), it is absent on other day types and not significant on expiry afternoons alone (p = 0.15). It was therefore pre-registered and tested again on fresh data (findings 8 and 9): it did not hold up.
6. **Out of sample, gamma at 14:30 does not improve last-hour forecasts on expiry days.** Over 155 test expiry days (17 August 2023 to 28 July 2026, closing-auction days excluded) adding it to a HAR-style model changes QLIKE loss by -1.8% (a small improvement), Diebold-Mariano p = 0.53. The test could only have detected an improvement of about 7.7% or more, so the sample is too small to rule out a small gain.
7. **A design fact that limits what this study can see.** Between 09:30 and 15:00, with at least 30 minutes left, the "reach" of each option's gamma (one standard deviation of the index move left before expiry) is on the median expiry day 166 index points at 09:30 and still 48 points at 15:00, about one 50-point strike spacing (`gamma_width_expiry_day.csv`). Gamma at spot is therefore a smooth function of the OI profile: it is almost uncorrelated with the distance to the nearest strike (within-day correlation between -0.008 and 0.005) and 0.97 correlated with its half-strike-shifted placebo on expiry days. The 5-minute time floor never binds on this grid. The reach falls clearly below the strike spacing only in the last minutes (34 points with 15 minutes left, 20 with 5), so the strike-by-strike gamma spikes that pinning theory is about exist only after 15:00, which this design leaves out (and which the closing auction changed from 3 August 2026). Finding 8 tests exactly that window.
8. **The sharpest test the data allow, the final half hour of expiry days, also finds nothing (pre-registered follow-up, Section 8).** On 187 expiry days before the closing auction, at every minute from 15:00 to 15:24, the variance of the next 5 minutes does not respond to gamma at spot: coefficient 0.257, 95% interval -0.327 to 0.841, p = 0.39, with the move since the previous close and the distances to the nearest and to the max-OI strike controlled. Here gamma at spot finally depends on the exact strikes (its within-day correlation with the half-strike-shifted placebo is 0.09, against 0.97 before 15:00), so the test can tell real strikes from a smooth profile; the placebos are null too. Gamma recomputed for NSE's averaged settlement (far sharper) gives 0.005 (p = 0.28). Concentration gives -0.424 (-1.910 to 1.062, p = 0.57), but in these minutes almost all gamma at spot already comes from within 0.5% of the index (median share 0.98 at 15:00, 1.00 at 15:24), so that measure barely varies and its test is weak.
9. **The concentration lead does not replicate on Bank Nifty (pre-registered follow-up, Section 9).** On all 98 Bank Nifty expiry days with weekly options (5 January 2023 to 13 November 2024), the identical specification gives -0.122, 95% interval -0.661 to 0.417, two-sided p = 0.66 (one-sided p = 0.33 for the predicted negative sign). The sign matches Nifty's, but the estimate is a third of the size, most fake profiles of placebos (b) and (c) give a |t| at least as large (randomization p = 0.87 and 0.74), and Holm over the three pre-registered tests gives adjusted p = 1.0 for each. The replication is weak: with 98 days it had about 29% power to detect an effect as large as Nifty's -0.386, and its interval contains both zero and the Nifty value. So this is not proof that the Nifty lead was false, but nothing now supports it.

**In one sentence:** minute-by-minute gamma at spot, built only from information available at that minute, does not move Nifty's volatility on expiry days in any way we can detect, not even in the final half hour where strike-level gamma is sharpest; the apparent effects on other days are the leverage effect in disguise; and the one measure that passed the placebos (how concentrated gamma is near spot) failed its pre-registered tests in the final half hour and on Bank Nifty.

## 1. The ideas in plain words

**Option, strike, expiry.** A Nifty call option with strike 24,000 pays the amount by which Nifty ends above 24,000 on its expiry day; a put pays the amount by which it ends below. NSE lists strikes every 50 points near the current level. Since September 2025 the weekly Nifty contract expires on Tuesday (Thursday before).

**Open interest (OI).** The number of option contracts that are open (bought by someone and sold by someone) at the end of a day. NSE publishes it per strike after the close in the bhavcopy (in shares of the index, in this project). During the next day it is not published again, so yesterday's OI is the latest public information.

**Gamma.** An option's delta is how much its price moves when the index moves one point. Gamma is how fast delta changes as the index moves. It is largest for strikes close to the current index level and, for those strikes, it grows as expiry approaches, because an option close to expiry flips quickly between "will pay" and "will not pay".

**Dealers and hedging.** Banks and trading firms that take the other side of customers' options ("dealers") usually hedge their delta by trading futures. If a dealer is **long gamma** (owns options, net), hedging means selling when the index rises and buying when it falls, which damps moves. If **short gamma**, hedging means buying rises and selling falls, which amplifies moves.

**Gamma exposure (GEX) at spot.** Add up, over all strikes, OI times gamma times the index level squared times 0.01: the rupee amount of delta that changes for a 1% index move. "Gross" GEX counts every option as positive, which needs no assumption about who holds it. "Signed" GEX multiplies each option by the dealers' assumed net share: +1 for calls and -1 for puts (the US convention), or weights from NSE's participant-wise OI for the "Pro" category, or the opposite of the "Client" category. NSE does not publish who holds which strike, so the sign is always a guess. **Concentration** is the share of gross GEX that comes from strikes within 0.5% of the index.

**Why intraday.** On expiry day the OI of the expiring contract is fixed from yesterday's close, but the index moves and the clock runs down, so GEX at spot changes minute by minute in a way anyone could compute in real time. Earlier project results used one GEX number per day and found nothing out of sample (`docs/REAL_DATA_RESULTS.md`, finding 4).

**Realized variance (RV).** The sum of squared 1-minute log returns over a window: a direct measure of how much the index moved. We use the next 15 minutes (and 30 as a check), on the log scale.

**Fixed effects.** A day fixed effect is a separate constant for each day; a minute-of-day fixed effect is a separate constant for each time of day. Including both means each anchor is compared only with other anchors of the same day (so news, VIX and anything else that is constant within a day cancel out) after removing the normal intraday pattern of volatility (high at the open, low at lunch, rising into the close). Since OI and implied vols are fixed within a day, and on expiry days the clock is the same function of the time of day for every day, what is left of gamma at spot varies with where the index sits relative to the OI profile (and, on other days, with how sharply the clock lets that profile show).

**Clustering.** Anchors from the same day share that day's shocks and overlap in their outcome windows, so they are not independent. Clustered standard errors treat each day as one unit of independent information, which makes intervals honest. We also report two-way clustering (by day and by time of day) and Driscoll-Kraay errors (which also allow neighbouring days to be related).

**Placebo.** A fake version of the explanatory variable that keeps everything except the part the theory says matters. If the fake gives the same result as the real one, the real result is not evidence for the theory.

**Revealed sign.** Gross GEX has no sign assumption, so the direction of its effect would itself say whether dealers are net long (more gamma, calmer) or net short (more gamma, wilder). That was the hope of this study.

## 2. Data (REAL DATA)

| Item | Value | Source file |
| --- | --- | --- |
| Panel days (3 Jan 2023 to 27 Oct 2026) | 937 | `data_checks.csv` |
| of which expiry days / day before expiry / ordinary days | 200 / 200 / 537 | `data_checks.csv` |
| Expiry days by regime: multi-weekly / one weekly Thursday / Tuesday | 98 / 41 / 61 | `regressions.csv` (exploratory rows) |
| Anchors per day (every 5 minutes, 09:30 to 15:00) | 67 | `results/intraday_gamma_stdout.txt` |
| Anchor rows | 62,779 | `data_checks.csv` |
| Days whose chain comes from the previous trading day (no look-ahead) | 937 of 937 | `data_checks.csv` |
| Days needing a holiday re-dated contract | 4 | `data_checks.csv` |
| Closing-auction days (from 3 Aug 2026) | 59 | `data_checks.csv` |
| Median strikes per chain within 10% of the forward | 85 | `data_checks.csv` |
| Median share of chain rows whose implied vol had to be filled from neighbours | 0.169 | `data_checks.csv` |
| Rows with no next-30-minute outcome (closing-auction windows) | 177 | `data_checks.csv` |

Two special sessions are left out because the minute bars do not cover them (1 November 2024, no bars; 21 October 2025, Muhurat session with 16% coverage).

**What gamma at spot looks like** (`descriptives_by_time.csv`, medians over days):

| Day type | Time | Gross GEX at spot (crore rupees per 1% move) | Concentration (share within 0.5%) | Cross-day SD of the within-day deviation of log GEX |
| --- | --- | --- | --- | --- |
| expiry | 09:30 | 97,600 | 0.515 | 0.108 |
| expiry | 12:00 | 96,712 | 0.605 | 0.053 |
| expiry | 14:30 | 91,331 | 0.904 | 0.104 |
| expiry | 15:00 | 90,101 | 0.982 | 0.153 |
| day before | 14:30 | 71,233 | 0.477 | 0.061 |
| ordinary | 14:30 | 42,503 | 0.336 | 0.051 |

(The exact figures in the file are 97,599.7, 96,712.3, 91,330.8 and so on; rounded here.) Gross GEX at spot does **not** explode towards the close on expiry day: what grows is its concentration.

**How wide one option's gamma is on expiry day** (`gamma_width_expiry_day.csv`; one standard deviation of the index move left before expiry, from each expiry day's previous-close at-the-money implied vol, in index points):

| Minutes to 15:30 | On the anchor grid? | Median width (points) | 10% to 90% of days | Median width / 50-point strike spacing |
| --- | --- | --- | --- | --- |
| 360 (09:30) | yes | 165.8 | 112.3 to 258.3 | 3.32 |
| 120 (13:30) | yes | 95.7 | 64.8 to 149.1 | 1.91 |
| 60 (14:30) | yes | 67.7 | 45.9 to 105.5 | 1.35 |
| 30 (15:00) | yes | 47.8 | 32.4 to 74.6 | 0.96 |
| 15 | no | 33.8 | 22.9 to 52.7 | 0.68 |
| 5 | no | 19.5 | 13.2 to 30.4 | 0.39 |

This is why the level does not explode: for a smooth OI profile, summing a narrowing gamma "bump" over many strikes gives roughly the OI density at the index level, whatever the time left; only when the bump becomes narrower than the strike spacing do individual strikes stand out.

## 3. Method

**Chain and implied vols.** For day t with next expiry E (E = t on expiry days), we take the contract expiring on E from the bhavcopy of the last trading day before t (published after that close). Implied vols are backed out of that day's settlement prices with the package's Black-76 solver at that day's time to expiry, sorted by strike, gaps filled from neighbouring strikes (`pinning_study.prepare_chain`), and held fixed per strike during day t ("sticky strike"). Strikes within 10% of the previous forward are kept.

**Anchors and timing.** Anchors are 09:30, 09:35, ..., 15:00 (67 per day). The index level at an anchor is the close of the last completed 1-minute bar (the bar stamped one minute earlier), so nothing after the anchor is used; a unit test checks this by changing every later bar.

**Time to expiry** in trading years: (max(minutes to 15:30, floor) + 375 x full sessions left) / (375 x 252). The forward inside the day is the index times exp(6.5% x T). The main floor is 5 minutes; since the last anchor has 30 minutes left it never binds (`data_checks.csv`). Sensitivity: floors of 60 and 120 minutes, and a clock that ends at 15:10 (the official close averages 15:00 to 15:30, and the variance of that average equals that of a single price at about 15:10).

**Variables.** Gross GEX at spot (rupees per 1% move, OI in shares), its concentration, and signed GEX under the three rules divided by gross GEX at the previous close (a day constant known before the open, so that days with very different OI are comparable). Controls: log RV of the previous 15 and 60 minutes (before 10:15, the 60-minute lag is the RV since 09:15 scaled to 60 minutes) and the absolute distance to the nearest 50-point strike. Extra variables for robustness: distance to the strike with the most OI, and the signed and absolute index move since the previous close.

**Outcome.** Log RV of the next 15 minutes (and 30). From 3 August 2026 the 1-minute index is frozen from 15:15 to about 15:28 because of NSE's closing auction, so windows that reach past 15:14 on those days are dropped.

**Regression.** Log RV(next 15 minutes) on log gross GEX at spot (or one signed version, or concentration) plus the controls, with day and minute-of-day fixed effects, run separately for expiry days, days before expiry and ordinary days. Standard errors clustered by day; two-way and Driscoll-Kraay p-values alongside.

**Pre-specified headline set** (fixed before any result was seen, from the study brief): the gross coefficient for each of the three day types, the three signed coefficients on expiry days, and the out-of-sample test (7 tests, Holm-adjusted together). Two things were **added after the first run** and are labelled so in `design_log.csv`: the leverage control (signed and absolute move since the previous close) and the placebos on the expiry-afternoon subsample. Concentration was computed from the start but was not a headline test. The two follow-ups of Sections 8 and 9 were written down in `design_log.csv` (rows starting "PRE-REGISTERED 2026-11-02 21:36 UTC") before any of their code was written or run.

## 4. Results

### 4.1 Headline (REAL DATA)

Source: `regressions.csv` (rows with `family = headline`) and `all_tests.csv` (Holm column).

| Day type | Regressor | Coefficient | 95% interval | p (clustered by day) | p two-way | p Driscoll-Kraay | Holm p | Days | Anchors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expiry | log gross GEX | 0.142 | -0.128 to 0.413 | 0.301 | 0.286 | 0.233 | 0.903 | 200 | 13,400 |
| day before | log gross GEX | -0.723 | -1.124 to -0.322 | 0.0005 | 0.0004 | 0.0010 | 0.003 | 200 | 13,400 |
| ordinary | log gross GEX | -0.579 | -1.044 to -0.113 | 0.015 | 0.016 | 0.013 | 0.090 | 537 | 35,979 |
| expiry | signed, US convention | -0.130 | -0.290 to 0.031 | 0.114 | 0.107 | 0.137 | 0.456 | 200 | 13,400 |
| expiry | signed, Pro | -3.70 | -11.70 to 4.30 | 0.363 | 0.390 | 0.318 | 0.903 | 200 | 13,400 |
| expiry | signed, Client mirror | 3.33 | 0.09 to 6.56 | 0.044 | 0.028 | 0.024 | 0.219 | 200 | 13,400 |

The seventh headline test (out of sample) is in Section 6. Signed coefficients are per unit of "net GEX at spot divided by gross GEX at the previous close", so their sizes differ between rules (Pro weights are small numbers); `regressions.csv` also gives each coefficient times the within-day standard deviation of its regressor (`effect_per_within_sd`): 0.012 for gross on expiry days, -0.019, -0.011 and 0.020 for the three signed versions.

### 4.2 Why the day-before and ordinary-day coefficients are not hedging

Source: `within_day_correlations.csv` (correlations after removing day and minute effects).

| Sample | Within-day SD of log gross GEX | Correlation with signed move since previous close | with absolute move | with distance to nearest strike | with the +25-point placebo |
| --- | --- | --- | --- | --- | --- |
| expiry | 0.083 | 0.296 | -0.573 | -0.002 | 0.973 |
| day before | 0.047 | 0.629 | -0.424 | 0.005 | 0.990 |
| ordinary | 0.039 | 0.722 | -0.584 | -0.003 | 0.997 |
| expiry 13:00 to 15:00 | 0.061 | 0.202 | -0.524 | -0.008 | 0.948 |

Within a day, gross GEX at spot goes up when the index goes up. Part of this is mechanical (the definition multiplies by the index level squared, so a 1% rise adds about 2%, which is half of a typical within-day swing on ordinary days); part comes from the shape of the OI profile. Volatility tends to fall after the index rises. Adding the signed and absolute move since the previous close removes the day-before and ordinary-day coefficients (`regressions.csv`, spec "plus signed and absolute move since previous close"):

| Sample | Coefficient with the move controls | 95% interval | p |
| --- | --- | --- | --- |
| expiry | 0.212 | -0.144 to 0.567 | 0.241 |
| day before | 0.266 | -0.506 to 1.039 | 0.497 |
| ordinary | -0.296 | -1.086 to 0.493 | 0.462 |
| expiry 13:00 to 15:00 | 0.361 | -0.269 to 0.992 | 0.260 |

### 4.3 Signed versions on every day type

Source: `regressions.csv`. "Main" is the pre-specified specification; "with move controls" adds the signed and absolute move since the previous close.

| Sample | Rule | Main coefficient (p) | With move controls (p) |
| --- | --- | --- | --- |
| expiry | US convention | -0.130 (0.114) | 0.113 (0.443) |
| expiry | Pro | -3.70 (0.363) | -4.63 (0.261) |
| expiry | Client mirror | 3.33 (0.044) | 0.74 (0.751) |
| day before | US convention | -0.845 (< 0.001) | -0.627 (0.064) |
| day before | Pro | 5.01 (0.473) | -10.16 (0.149) |
| day before | Client mirror | 12.24 (< 0.001) | -0.94 (0.785) |
| ordinary | US convention | -0.537 (< 0.001) | 0.210 (0.530) |
| ordinary | Pro | 9.81 (0.050) | -4.49 (0.417) |
| ordinary | Client mirror | 8.80 (< 0.001) | 0.72 (0.821) |
| expiry 13:00 to 15:00 | US convention | -0.284 (0.105) | -0.262 (0.396) |
| expiry 13:00 to 15:00 | Pro | -7.81 (0.296) | -12.08 (0.135) |
| expiry 13:00 to 15:00 | Client mirror | 4.76 (0.132) | -0.93 (0.846) |

Under the US convention, net GEX is call gamma minus put gamma; most call OI sits above the index and most put OI below, so it tends to rise when the index rises. That is why the US-convention and Client-mirror coefficients on non-expiry days look so strong and then disappear.

### 4.4 Secondary results and robustness of the expiry-day null

Source: `regressions.csv`.

| Sample and specification | Coefficient | 95% interval | p |
| --- | --- | --- | --- |
| expiry, next 30 minutes | 0.140 | -0.156 to 0.437 | 0.351 |
| day before, next 30 minutes | -0.713 | -1.112 to -0.314 | 0.0005 |
| ordinary, next 30 minutes | -0.446 | -0.922 to 0.030 | 0.066 |
| expiry, time floor 60 minutes | 0.132 | -0.145 to 0.409 | 0.349 |
| expiry, time floor 120 minutes | 0.123 | -0.169 to 0.415 | 0.408 |
| expiry, clock ends 15:10 | 0.133 | -0.112 to 0.378 | 0.285 |
| expiry, without the distance-to-strike control | 0.142 | -0.129 to 0.413 | 0.303 |
| expiry, distance bins x hour dummies instead | 0.148 | -0.122 to 0.417 | 0.282 |
| expiry, plus distance to max-OI strike and the move controls | 0.219 | -0.203 to 0.642 | 0.308 |
| expiry, before the closing auction (187 days) | 0.117 | -0.161 to 0.395 | 0.406 |
| expiry, multi-weekly regime (98 days, exploratory) | 0.141 | -0.284 to 0.565 | 0.512 |
| expiry, Thursday single-weekly regime (41 days, exploratory) | 0.017 | -0.567 to 0.601 | 0.953 |
| expiry, Tuesday regime (61 days, exploratory) | 0.271 | -0.230 to 0.772 | 0.283 |
| expiry 13:00 to 15:00 | 0.534 | 0.046 to 1.022 | 0.032 |
| expiry 13:00 to 15:00, time floor 60 / 120 minutes | 0.517 / 0.594 | 0.002 to 1.033 / 0.005 to 1.183 | 0.049 / 0.048 |
| expiry, concentration instead of level | -0.386 | -0.727 to -0.044 | 0.027 |
| expiry, concentration, with move controls | -0.450 | -0.793 to -0.106 | 0.010 |
| expiry 13:00 to 15:00, concentration | -0.425 | -1.012 to 0.162 | 0.155 |
| day before, concentration | -0.003 | -0.324 to 0.317 | 0.984 |
| ordinary, concentration | 0.142 | -0.066 to 0.351 | 0.181 |

The time floor makes almost no difference because on this grid it rarely binds; the 60- and 120-minute floors only cap T for the last 6 and 18 anchors.

## 5. Placebos (REAL DATA)

Source: `placebo_summary.csv` (draws in `placebo_draws.csv`). Each placebo recomputes gamma at spot with a fake OI profile and reruns the same regression.

* **(a) Half-strike shift.** Every strike of the OI profile moved by +25 points (and, as a check, -25), so OI no longer sits on real strikes. A hedging effect tied to the real strikes should vanish; a pure "near a round number" effect would not.
* **(b) Another day's profile.** The OI of a randomly chosen other day of the same type, moved by a whole number of 50-point strikes so that its at-the-money point sits on today's, with today's implied vols. Strikes stay on the real grid, but the OI shape is wrong. 200 draws.
* **(c) Permuted OI.** The (call OI, put OI) pairs randomly permuted across the strikes within 5% of the previous forward; strikes and implied vols stay. 200 draws.

For (b) and (c) the table gives the median coefficient over draws, the 2.5% to 97.5% range of draws, and a randomization p-value: the share of draws whose |t| is at least the real |t|. A small randomization p means the real profile does something the fake ones do not.

**Gross GEX at spot:**

| Sample | Real coefficient (p) | (a) +25 shift (p) | (a') -25 shift (p) | (b) median [range], randomization p | (c) median [range], randomization p |
| --- | --- | --- | --- | --- | --- |
| expiry | 0.142 (0.301) | 0.089 (0.505) | 0.173 (0.230) | 0.016 [-0.192, 0.216], 0.189 | -0.098 [-0.255, 0.064], 0.547 |
| day before | -0.723 (0.0005) | -0.821 (< 0.001) | -0.583 (0.014) | -0.801 [-1.097, -0.539], 0.706 | -0.413 [-1.027, 0.189], 0.010 |
| ordinary | -0.579 (0.015) | -0.585 (0.009) | -0.566 (0.026) | -0.576 [-0.782, -0.368], 0.547 | -0.992 [-1.486, -0.494], 0.746 |
| expiry 13:00 to 15:00 | 0.534 (0.032) | 0.392 (0.107) | 0.601 (0.018) | 0.336 [-0.019, 0.664], 0.070 | -0.078 [-0.304, 0.198], 0.055 |

On the day before expiry, 98.0% of the (b) draws are significant at 5%; on ordinary days 81.5% of (b) and 87.0% of (c) draws are. The fake profiles reproduce the effect, which is what one expects if it comes from "the index moving away from where yesterday's options are concentrated" (any realistic profile is centred near yesterday's close) together with the leverage effect, and not from the real positions. On expiry days nothing is there to reproduce.

A joint regression with the real and the +25-shifted gamma (`regressions.csv`, spec "horse race") gives, on expiry days, 1.007 (p = 0.12) for the real and -0.865 (p = 0.17) for the shifted one; on expiry afternoons 1.573 (p = 0.046) and -1.083 (p = 0.16). Because the two variables are 0.95 to 0.97 correlated, these coefficients are fragile; on other days the same regression gives large, opposite, significant pairs (7.32 and -7.70 on the day before expiry) that only reflect the tiny difference between two almost identical variables.

**Concentration of gamma (share within 0.5% of spot):**

| Sample | Real coefficient (p) | (a) +25 shift (p) | (a') -25 shift (p) | (b) median [range], randomization p | (c) median [range], randomization p |
| --- | --- | --- | --- | --- | --- |
| expiry | -0.386 (0.027) | -0.021 (0.909) | -0.164 (0.319) | -0.081 [-0.257, 0.113], 0.010 | 0.010 [-0.167, 0.184], 0.025 |
| day before | -0.003 (0.984) | 0.134 (0.387) | 0.373 (0.033) | -0.044 [-0.240, 0.158], 0.980 | -0.025 [-0.358, 0.277], 0.995 |
| ordinary | 0.142 (0.181) | -0.067 (0.539) | 0.291 (0.009) | 0.085 [-0.058, 0.237], 0.219 | -0.002 [-0.211, 0.201], 0.174 |
| expiry 13:00 to 15:00 | -0.425 (0.155) | -0.019 (0.955) | -0.429 (0.181) | -0.423 [-0.760, -0.144], 0.582 | -0.109 [-0.460, 0.282], 0.204 |

On expiry days the concentration effect is specific to the real profile: the half-strike shift kills it, and the fake profiles of (b) and (c) rarely produce a |t| as large (randomization p = 0.010 and 0.025). Note that the concentration of the shifted profile is almost unrelated to the real one (within-day correlation 0.05), so this placebo is far more discriminating for concentration than for the level. On the other two day types, and on the afternoon alone, there is nothing specific to the real profile.

**Figure 1** (`fig1_deciles_expiry_afternoon.png`, data in `fig_deciles_expiry_afternoon.csv`): expiry afternoons, average next-15-minute RV by decile of gross GEX at spot, for the real profile and the +25 shift. Left: raw deciles, where RV falls steeply with GEX; this is a comparison across days, and days with high implied volatility have both lower gamma (gamma is inversely proportional to volatility) and higher RV. Right: after removing day and time-of-day means, the two lines lie on top of each other, with a mild rise in the top two deciles. The error bars treat anchors as independent and are therefore too narrow.

**Figure 2** (`fig2_coefficients_and_placebos.png`): the gross coefficients with the three placebos for each sample.

## 6. Out of sample (REAL DATA)

Source: `oos_summary.csv` (forecasts in `oos_forecasts.csv`, inputs in `oos_expiry_day_features.csv`).

Target: realized variance of the last hour (14:30 to 15:30) on expiry days. Base model (HAR style, on logs): RV so far today, RV of the 30 minutes before 14:30, yesterday's RV, the average RV of the last 5 days (daily RV measured 09:15 to 15:15 on every day so that closing-auction days are comparable) and yesterday's India VIX. The second model adds log gross GEX at spot at 14:30. Forecasts are exp(fitted value) times the mean of exp(residuals) in the training data. The models are refit every 21 trading days on all earlier expiry days; the first 150 trading days are training only. Loss: QLIKE. Test: Diebold-Mariano with the Harvey, Leybourne and Newbold correction (`niftygex.evaluation`).

| Specification | Test days | QLIKE, base | QLIKE, with gamma | Change | DM p (two-sided) | Smallest detectable change (80% power) | Days gamma model better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Main: last hour, closing-auction days excluded | 155 | 0.1805 | 0.1772 | -1.8% | 0.532 | about 7.7% | 42.6% |
| 14:30 to 15:15, all expiry days | 168 | 0.1888 | 0.1909 | +1.1% | 0.462 | about 4.0% | 41.1% |
| Main target, gamma relative to the previous close's gross GEX | 155 | 0.1805 | 0.1819 | +0.8% | 0.648 | about 5.5% | 47.7% |

The sample is too small to detect improvements of a few percent; it can only say that gamma at 14:30 does not give a large, reliable gain. The "smallest detectable change" is 2.8 standard errors of the mean loss difference, in percent of the base QLIKE.

## 7. Multiple testing

`all_tests.csv` lists every test run with its coefficient, interval, raw p, Holm-adjusted p over the headline set and Holm-adjusted p within its own family: 117 in the main study (7 headline, 19 secondary, 3 exploratory, 48 robustness, 40 placebo) and 34 in the follow-ups (3 pre-registered primary, 10 secondary, 1 robustness, 20 placebo; families starting `followup_`). The Holm adjustment over the three follow-up primary tests is in Sections 8 and 9. Three things in the follow-ups were not pre-registered and are labelled as such: the Bank Nifty log gross regression with the move controls, the horse-race regressions of real and shifted gamma (4 of the 20 follow-up placebo tests, as in the main study), and the power figures of Section 9, computed after the results were seen.

| Headline test | Raw p | Holm p |
| --- | --- | --- |
| expiry, log gross GEX | 0.301 | 0.903 |
| day before, log gross GEX | 0.0005 | 0.003 |
| ordinary, log gross GEX | 0.015 | 0.090 |
| expiry, signed (US convention) | 0.114 | 0.456 |
| expiry, signed (Pro) | 0.363 | 0.903 |
| expiry, signed (Client mirror) | 0.044 | 0.219 |
| out of sample, Diebold-Mariano | 0.532 | 0.903 |

The one headline test that survives Holm (day before expiry) is the one that Sections 4.2 and 5 show to be the leverage effect rather than hedging.

## 8. Follow-up 1: the final half hour of expiry days (pre-registered, REAL DATA)

**Why.** Finding 7 showed that before 15:00 each option's gamma is wider than the strike spacing, so gamma at spot is a smooth function of the OI profile and cannot show strike-by-strike hedging. Only in the last minutes does the gamma of single strikes stand out, which is where pinning theory says hedging should matter most. Before the closing auction (to 31 July 2026) the index trades normally from 15:00 to 15:29: the share of 1-minute returns that are exactly zero is 0.37% in that half hour against 0.39% from 14:15 to 14:59, and one bar in 15:00 to 15:29 is missing over all sample days (`followup_data_checks.csv`).

**What was pre-registered** (`design_log.csv`, written at 21:36 UTC on 2 November 2026, before any follow-up code; the main study's results, including the concentration lead and the gamma widths, had been seen; the final half hour's volatility had entered the main study only through the outcome windows of the last anchors, which reach into 15:00 to 15:29, and the out-of-sample last-hour target; no final-half-hour gamma had been computed):

| Item | Choice |
| --- | --- |
| Sample | Nifty expiry days before 3 August 2026: 187 days, 5 January 2023 to 28 July 2026 |
| Anchors | every minute 15:00 to 15:24 (25 per day, 4,675 rows); index level = close of the last completed bar |
| Outcome | log realized variance of the next 5 minutes (no zero windows) |
| Primary test FH1 | log gross GEX at spot, clock to 15:30, 1-minute floor; no predicted sign |
| Primary test FH2 | concentration (share within 0.5% of spot), same clock; predicted negative |
| Controls | log RV of the previous 5, 15 and 60 minutes, distance to the nearest 50-point strike, distance to the max-OI strike, signed and absolute move since the previous close |
| Fixed effects and errors | day and minute of day; clustered by day (two-way and Driscoll-Kraay alongside); two-sided p |
| Secondary | floors of 2 and 5 minutes; gamma for the averaged settlement with floors 1, 2 and 5; concentration within 0.25%; main-study controls only |
| Placebos | (a) profile shifted +25 and -25 points, (b) another final-half-hour day's profile re-centred, (c) OI permuted within 5%; 200 draws each |
| Multiple testing | Holm over FH1, FH2 and BN1 (Section 9) |

**Two versions of the clock.** (i) Minutes to 15:30, as in the main study. At 15:24 six minutes remain, so floors of 1, 2 and 5 minutes never bind and the three versions are identical (`followup_data_checks.csv`); the pre-registration said so in advance. (ii) **Averaged settlement.** Before the closing auction NSE settles the options on the official close, which averages prices over 15:00 to 15:30; we model it as the mean of the 30 one-minute closes of the 15:00 to 15:29 bars. At an anchor with n of those closes still to come, a share w = n/30 of the average is still open and the rest is the mean of the closes already printed (known at the anchor). Each option is then w times an option on the average of the remaining closes, with strike (K - (1 - w) x printed mean) / w, and that average has the variance of one price (n + 1)(2n + 1)/(6n) minutes ahead: 10.5 minutes at 15:00 and 2.5 at 15:24. Seen from the open part of the average, the strikes are 1/w times further apart, so this gamma is far sharper: its median across days falls from 88,988 crore rupees per 1% at 15:00 to 22 crore at 15:24, while the median of the clock-to-15:30 version stays between about 84,000 and 88,000 (`followup_descriptives.csv`). The 5-minute floor binds for this version from 15:17 (8 anchors a day); the 2-minute floor never does.

**What the variables look like** (`followup_descriptives.csv`, medians over the 187 days):

| Time | Gross GEX at spot, clock to 15:30 (crore per 1%) | Gross GEX at spot, averaged settlement | Concentration within 0.5% | RV of the next 5 minutes (x 1e6) | Distance to nearest strike (points) |
| --- | --- | --- | --- | --- | --- |
| 15:00 | 86,950 | 88,988 | 0.982 | 0.589 | 10.8 |
| 15:06 | 86,154 | 57,674 | 0.993 | 0.399 | 10.05 |
| 15:12 | 87,930 | 32,357 | 0.998 | 0.241 | 11.1 |
| 15:18 | 84,210 | 10,333 | 1.000 | 0.202 | 11.3 |
| 15:24 | 84,419 | 22 | 1.000 | 0.157 | 11.1 |

**Results** (`followup_regressions.csv`; primary rows also in `followup_summary.csv`):

| Specification | Coefficient | 95% interval | p | p two-way | p Driscoll-Kraay |
| --- | --- | --- | --- | --- | --- |
| **FH1**: log gross GEX at spot (primary) | 0.257 | -0.327 to 0.841 | 0.386 | 0.355 | 0.371 |
| **FH2**: concentration within 0.5% (primary) | -0.424 | -1.910 to 1.062 | 0.574 | 0.600 | 0.468 |
| log gross, floor 2 or 5 minutes (identical to FH1) | 0.257 | -0.327 to 0.841 | 0.386 | 0.355 | 0.371 |
| log gross, averaged settlement, floor 1 (floor 2 identical) | 0.0049 | -0.0041 to 0.0140 | 0.282 | 0.273 | 0.291 |
| log gross, averaged settlement, floor 5 | 0.0105 | -0.0077 to 0.0288 | 0.256 | 0.254 | 0.253 |
| concentration within 0.25% | -0.118 | -0.585 to 0.350 | 0.621 | 0.598 | 0.587 |
| log gross, main-study controls only | 0.256 | -0.360 to 0.872 | 0.414 | 0.390 | 0.383 |
| concentration, main-study controls only | -0.549 | -2.064 to 0.967 | 0.476 | 0.487 | 0.381 |

How large could an effect be? For FH1, one within-day standard deviation of log gross GEX at spot is 0.091 (`within_sd_x`), so the interval allows anything from about 3% less to about 8% more next-5-minute variance per such swing; the smallest coefficient the test could detect with 80% power is 0.83 (`mde_80pct_power`). The averaged-settlement version varies enormously within a day (within-day SD of its log 4.5), so its small coefficient corresponds to an effect per standard deviation of 0.022 (interval about -0.02 to 0.06). For FH2, concentration within 0.5% hardly moves in these minutes (within-day SD 0.029, against 0.053 in the main study), so the test is weak (smallest detectable coefficient 2.1); the narrower 0.25% band varies more (SD 0.074) and gives nothing either.

**Placebos** (`followup_placebo_summary.csv`, draws in `followup_placebo_draws.csv`):

| Regressor | Real coefficient (p) | (a) +25 shift (p) | (a') -25 shift (p) | (b) median [range], randomization p | (c) median [range], randomization p |
| --- | --- | --- | --- | --- | --- |
| log gross GEX (FH1) | 0.257 (0.386) | -0.099 (0.767) | -0.342 (0.271) | 0.019 [-0.414, 0.438], 0.363 | 0.017 [-0.220, 0.288], 0.388 |
| concentration (FH2) | -0.424 (0.574) | -0.641 (0.387) | -0.415 (0.606) | -0.630 [-1.065, -0.216], 0.801 | -0.597 [-1.142, 0.166], 0.751 |

In this window the half-strike placebo is finally a sharp test for the level of gamma: the shifted profile's gamma at spot has a within-day correlation of only 0.09 (+25) and -0.005 (-25) with the real one, against 0.97 before 15:00. So the design can now tell real strikes from a smooth profile, and the real strikes still show nothing. For concentration the fake profiles of (b) and (c) give coefficients as negative as the real one (within-day correlation with the real concentration 0.85 and 0.73): in the final half hour, concentration is set mostly by the clock and by where the index sits between strikes, not by the shape of the OI profile.

**Verdict.** No sign of hedging in the final half hour. Neither primary test is significant (Holm-adjusted p = 1.0 for both, `followup_summary.csv`), the sharp averaged-settlement gamma shows nothing, and the placebos behave like the real profile.

**Figure 3** (`fig3_followups.png`): the primary coefficients of both follow-ups with their placebos, and the Nifty main-study concentration estimate for comparison.

## 9. Follow-up 2: does the concentration lead replicate on Bank Nifty? (pre-registered, REAL DATA)

**Why Bank Nifty.** It has its own option book, its own dealers' positions and, until November 2024, its own weekly expiry, so it is an independent check of finding 5. The specification is the Nifty secondary test, unchanged except for the 100-point strike spacing; it was written down before the Bank Nifty code was written (`design_log.csv`).

| Item | Value | Source |
| --- | --- | --- |
| Sample | every BANKNIFTY expiry date in the bhavcopies from 3 January 2023 to 19 November 2024 (weekly options stopped on 20 November 2024): 98, none left out | `followup_data_checks.csv` |
| First and last day | 5 January 2023 to 13 November 2024 | `followup_data_checks.csv` |
| Weekday of expiry | Wednesday 56, Thursday 38, Tuesday 4 | `followup_data_checks.csv` |
| Chain | expiring BANKNIFTY contract from the previous trading day's bhavcopy (98 of 98; 1 holiday re-dated contract); median 77 strikes within 10%; median spacing near the money 100 points | `followup_data_checks.csv`, `followup_banknifty_day_inputs.csv` |
| Bars | `data/raw/banknifty_1min.csv`; official closes "Nifty Bank" in `index_daily.csv` | |
| Anchors | every 5 minutes, 09:30 to 15:00: 6,566 rows, no zero RV windows | `followup_data_checks.csv` |
| BN1 (primary) | log RV of the next 15 minutes on concentration (share within 0.5% of spot), controls log RV of the previous 15 and 60 minutes and distance to the nearest 100-point strike, day and minute fixed effects, 5-minute floor, clustered by day; predicted negative; two-sided p primary, one-sided reported | `design_log.csv` |

Concentration follows the same daily path as on Nifty (median 0.44 at 09:30, 0.55 at 12:00, 0.83 at 14:30, 0.95 at 15:00; Nifty 0.515, 0.605, 0.904, 0.982), and its within-day standard deviation is the same (0.054 against 0.053), so the two tests measure the same thing (`followup_descriptives.csv`, `followup_regressions.csv`, `regressions.csv`).

**Results** (`followup_regressions.csv`, `followup_summary.csv`):

| Specification | Coefficient | 95% interval | p | p two-way | p Driscoll-Kraay |
| --- | --- | --- | --- | --- | --- |
| **BN1**: concentration (primary) | -0.122 | -0.661 to 0.417 | 0.655 (one-sided 0.328) | 0.655 | 0.643 |
| concentration, plus signed and absolute move since previous close | -0.105 | -0.644 to 0.434 | 0.700 | 0.701 | 0.682 |
| log gross GEX at spot | -0.109 | -0.408 to 0.190 | 0.471 | 0.446 | 0.337 |
| log gross, plus the move controls (added, not pre-registered) | -0.176 | -0.621 to 0.269 | 0.434 | 0.422 | 0.315 |
| For comparison, Nifty main study, concentration (same specification) | -0.386 | -0.727 to -0.044 | 0.027 | | |

**Placebos** (`followup_placebo_summary.csv`):

| Regressor | Real coefficient (p) | (a) +50 shift (p) | (a') -50 shift (p) | (b) median [range], randomization p | (c) median [range], randomization p |
| --- | --- | --- | --- | --- | --- |
| concentration (BN1) | -0.122 (0.655) | -0.291 (0.183) | -0.129 (0.628) | 0.290 [0.000, 0.620], 0.871 | 0.153 [-0.123, 0.505], 0.736 |
| log gross GEX | -0.109 (0.471) | -0.098 (0.501) | -0.114 (0.469) | -0.129 [-0.354, 0.099], 0.637 | -0.123 [-0.378, 0.172], 0.617 |

**Holm over the three pre-registered tests** (`followup_summary.csv`): FH1 raw p 0.386, FH2 0.574, BN1 0.655; Holm-adjusted p 1.0 for each.

**Verdict.** The lead did not replicate. The Bank Nifty coefficient has the predicted sign but is a third of Nifty's (-0.122 against -0.386; per within-day standard deviation -0.007 against -0.020), it is far from significant, the +50-point shifted profile gives a more negative coefficient than the real one (-0.291), and the fake profiles of (b) and (c) mostly give a larger |t|. The test is not strong: with 98 days its standard error is 0.27, so it had about 29% power (two-sided; 41% one-sided) to detect an effect as large as Nifty's (`power_vs_nifty_coef_two_sided`, normal approximation), and its interval contains both zero and -0.386. Together with the final-half-hour result, the honest reading is that the Nifty concentration result was most likely one of the false positives to be expected among 117 tests, or a real effect too small for these samples; nothing in the follow-ups supports it.

## 10. Limitations

* **Sticky-strike implied vols.** Each strike keeps yesterday's implied vol all day. Real implied vols move with the index and change shape through expiry day; gamma near the money is inversely proportional to implied vol, so this matters most late on expiry day.
* **End-of-day OI only.** OI changes during the day (especially on expiry day, when positions are closed or rolled), and NSE publishes it only after the close. Gamma at spot is built from yesterday's positions.
* **Third-party minute bars.** The 1-minute index bars come from a public GitHub dataset (`data/raw/minute_bars_source.json`) that the project checked against NSE's daily highs and lows, not from NSE itself. From 3 August 2026 the bars are frozen from 15:15 because of the closing auction.
* **The dealer sign is proxied.** NSE does not publish who holds which strike. The three signed rules are guesses, and the participant weights are market-wide for all index options, the same at every strike.
* **What the grid can see.** With anchors from 09:30 to 15:00, at least 30 minutes remain, so gamma at spot is smooth across strikes and the half-strike placebo is weak for the level of gamma (correlation 0.95 to 1.00 with the real one). The last 30 minutes, where strike-level gamma is sharpest, are covered by the follow-up of Section 8, but only before the closing auction (187 days).
* **Forward and rate.** The forward inside the day is the index times exp(6.5% x T), not a market forward; on expiry day the difference is under 6 index points.
* **Choices made after the first run.** The leverage control and the expiry-afternoon placebos were added after seeing the first results (`design_log.csv`); the headline specification was not changed.
* **Power.** About 200 expiry days and an out-of-sample test with 155 days cannot detect small effects. The expiry-day interval rules out elasticities outside -0.13 to 0.41, not small ones. The Bank Nifty replication has 98 days and about 29% power against an effect of the Nifty size.
* **The averaged settlement is modelled, not observed.** NSE's official close is built from each constituent's prices over 15:00 to 15:30, not from the index's one-minute closes; we use the mean of the 30 one-minute index closes and a driftless random-walk formula for the time left. In the final minutes implied vols are also least reliable, and they are held at yesterday's level.
* **Concentration saturates in the final half hour.** With so little time left almost all gamma at spot comes from within 0.5% of the index, so FH2 is a weak test; the 0.25% band is the better measure there and is null too.
* **Bank Nifty specifics.** The same sticky-strike vols and end-of-day OI; participant weights are market-wide and were not used; and Bank Nifty changed its expiry weekday during the sample (Thursday, then Wednesday, with a few Tuesdays), which the day fixed effects absorb but which may change who trades the last day.

## 11. Reproduce

```bash
OMP_NUM_THREADS=1 python experiments/intraday_gamma.py      # about 2.5 to 4 minutes on 4 cores, follow-ups included
OMP_NUM_THREADS=1 python -m pytest -q tests/test_intraday_gamma.py
```

Outputs in `results/intraday_gamma/`: `anchor_panel.csv.gz` (every anchor with all variables in levels; the regressions use their natural logs), `day_inputs.csv` (one row per day: chain date, forward, weights, scale), `data_checks.csv`, `descriptives_by_time.csv`, `gamma_width_expiry_day.csv`, `within_day_correlations.csv`, `regressions.csv`, `placebo_summary.csv`, `placebo_draws.csv`, `oos_summary.csv`, `oos_forecasts.csv`, `oos_expiry_day_features.csv`, `all_tests.csv`, `design_log.csv`, `fig_deciles_expiry_afternoon.csv` and the first two figures. Follow-ups: `followup_summary.csv` (the three pre-registered tests with Holm, placebos and power), `followup_regressions.csv`, `followup_placebo_summary.csv`, `followup_placebo_draws.csv`, `followup_data_checks.csv`, `followup_descriptives.csv`, `followup_banknifty_day_inputs.csv`, `followup_final_half_hour_panel.csv.gz`, `followup_banknifty_panel.csv.gz` and `fig3_followups.png`. The script reuses the chain helpers of `experiments/pinning_study.py` and the package's Black-76 greeks, GEX sign weights and Diebold-Mariano test.
