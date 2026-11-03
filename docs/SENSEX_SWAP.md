# Does the expiring index lead? Nifty, Sensex and the expiry-weekday swap of 1 September 2025 (P92)

**Status (3 November 2026).** Every result here is **REAL DATA**: third-party one-minute bars of Nifty 50 and Sensex,
Nifty expiry days from NSE bhavcopies, and the reported Sensex expiry calendar. Nothing is simulated except in the unit
tests (`tests/test_sensex_swap.py`, synthetic data only). The design was pre-registered in
[prereg/P92_SENSEX_SWAP.md](prereg/P92_SENSEX_SWAP.md) and committed to git as commit 042d510 at
2026-11-03T00:25:04Z, before any connectedness, spillover or volatility of this study was computed. The analysis was
then run with the committed script, unchanged (`results/sensex_swap/run_info.json`, `prereg_commit`). Every number
below is copied, rounded, from a file in `results/sensex_swap/`, named under each table; a few differences of two cell
means are worked out in the text from the numbers shown.

## The short answer

* **Primary test (H1): not supported.** Over the whole trading day, the expiring index did not become the net
  transmitter. The pre-registered difference in differences is **+0.42 points** (HAC 95% interval -1.83 to +2.67),
  **permutation p = 0.33** (one-sided). By the pre-stated rule the verdict is **"not supported; effects as large as the
  pre-registered minimum detectable effect (4.90 points) are ruled out"**, because the one-sided 95% upper bound is
  2.30 points.
* **But the design check failed.** At a fake swap date inside the 2025 window, where nothing changed, the same
  comparison moved by +4.45 points (p = 0.038). On Mondays and Wednesdays, where no index expires, it moved by
  -4.96 points around the real swap (p = 0.005). So the gaps between weekdays in this measure drift by several points
  for reasons that have nothing to do with which index expires. That is about as large as the effect we were looking
  for, so even the "ruled out" statement is weaker than it sounds.
* **Secondary tests (Holm-corrected):** in the **last hour** (14:15 to 15:14) the shift is in the H1 direction and
  survives the correction: +2.51 points, p = 0.005, Holm p = 0.015. On either index's expiry day, total connectedness
  is not clearly different from other days (-1.01 points, Holm p = 0.40). At the January 2025 Sensex move, the sign is
  the wrong way (+0.81, Holm p = 0.60).
* **Robustness:** the whole-day null result holds for other block lengths, without closing-auction days, without
  monthly expiry days, with one volatility fit, with week fixed effects, with ranks and with the paper's pooled VAR.
  Two other settings point the H1 way: the intraday part of volatility as input (+4.71, p = 0.015) and a 10-lag VAR
  at a 30-minute horizon (+3.05, p = 0.008). These checks were pre-stated as not decisive.
* **Plain reading:** there is no reliable evidence that the expiring index leads the other over the whole day. There
  is a sign of it in the last hour, which needs a fresh test on new data, especially because the design check failed.

## 1. Words used here

* **Volatility:** how much an index moves. Here, the expected size of the next one-minute move from the mcsGARCH model
  (the anchor paper's model), built exactly as in our Nifty and Bank Nifty connectedness study
  ([CONNECTEDNESS_GAMMA.md](CONNECTEDNESS_GAMMA.md)).
* **Spillover and connectedness (Diebold and Yilmaz 2012).** For each trading day, a small model (a VAR) predicts each
  index's volatility from the last two minutes of both indices. We then ask: of the error in forecasting Sensex
  volatility 10 minutes ahead, what share is due to surprises in Nifty? That is the **Nifty to Sensex share**. The
  other way round is the **Sensex to Nifty share**. Their sum is **total connectedness** (0 to 200, the paper's scale).
* **NET** = Nifty to Sensex share minus Sensex to Nifty share, in percentage points. Positive means Nifty "led" more
  than it "followed" that day: it was the **net transmitter**.
* **Expiry day:** the day on which an index's weekly options end. From January to August 2025 Sensex expired on
  Tuesday and Nifty on Thursday; from 1 September 2025 Nifty expires on Tuesday and Sensex on Thursday.
* **Difference in differences (delta):** the change in Tuesday's average NET from before to after the swap, minus the
  same change for Thursday. Anything special about Tuesdays in general, and anything that changed for every day at
  once, cancels out. If the expiring index leads, delta is positive.
* **Permutation test:** pretend many times (here 9,999) that the swap happened in a random set of weeks (moved in
  blocks of 4 weeks), and count how often a pretend swap gives a delta at least as large as the real one. That share
  is the p-value. Small means "unlikely to be chance".
* **HAC interval:** an interval from an ordinary regression whose standard errors allow neighbouring days to be alike
  (Newey and West). Reported next to the permutation test.
* **Holm correction:** when several secondary tests are run, Holm raises their p-values so that one lucky result is
  not mistaken for a finding.
* **Placebo:** the same test where no effect should exist. If a placebo "finds" something, the design is suspect.
* **Minimum detectable effect (MDE):** the smallest true delta the test would catch 80% of the time, fixed before
  the data were looked at.

## 2. The design in brief

Everything here was fixed in the pre-registration. On 1 September 2025 the two exchanges swapped expiry weekdays, so
the same weekday changed from one index's expiry day to the other's. The primary statistic is

    delta = [NET(Tue, after) - NET(Tue, before)] - [NET(Thu, after) - NET(Thu, before)],

with "before" = 1 January to 31 August 2025 and "after" = 1 September 2025 to 2 November 2026 (the last day with bars
for both indices). H1: delta > 0, one-sided, judged by the block permutation p-value at 0.05. Settings as in the
connectedness study: the 360-minute session 09:15 to 15:14 (so the frozen closing-auction minutes from 3 August 2026
never enter), one VAR(2) per day, horizon 10 minutes, mcsGARCH fitted separately on NSE's two expiry regimes.

## 3. Data and sample

480 trading days from 21 November 2024 to 2 November 2026 have bars for both indices, with no missing minute in the
session used; only the Diwali Muhurat session of 21 October 2025 is left out (`run_info.json`, `coverage.csv`). The
Sensex expiry calendar comes from the reported weekday rules and agrees with the 102 dates used in
[SENSEX_CONFOUND.md](SENSEX_CONFOUND.md) (`calendar_check.csv`). The input files are the same, byte for byte, as when
the coverage was counted before the pre-registration (`run_info.json` and `run_info_coverage.json` list the same
SHA-256 digits).

The mcsGARCH fits behave as in the connectedness study: intraday persistence between 0.990 and 0.997 for both indices in
both regimes (`mcsgarch_fits.csv`). The fast per-day connectedness code equals the replication's own function to
within 4e-12 points (`run_info.json`).

Nifty and Sensex are very tightly linked. On whole days, average total connectedness in the four H1 cells is between
91.35 and 93.76 on the paper's 0 to 200 scale (`cell_means.csv`), against an average per-day value of 79.0 for Nifty and
Bank Nifty (`results/connectedness_gamma/day_level_regression.csv`).

## 4. Primary test (H1, whole day)

**Table 1. Average NET in the four cells (REAL DATA, `cell_means.csv`, test P1).** NET in percentage points; positive
means Nifty led.

| Cell | Which index expires | Days | Mean NET | SD | Median |
| --- | --- | --- | --- | --- | --- |
| Tuesday, before | Sensex | 34 | +3.60 | 3.85 | +2.75 |
| Thursday, before | Nifty | 33 | +4.20 | 4.20 | +3.85 |
| Tuesday, after | Nifty | 56 | -1.53 | 2.47 | -1.21 |
| Thursday, after | Sensex | 56 | -1.35 | 2.63 | -1.26 |

Both weekdays fell by about 5 points from before to after (Tuesday by 5.14, Thursday by 5.55): NET was positive on
every kind of day in 2025 and slightly negative after the swap, a change common to all days that the design removes.
What is left is the difference between the two falls.

**Table 2. The primary test (REAL DATA, `tests.csv`, row P1; `verdict.json`).**

| | Value |
| --- | --- |
| delta (difference in differences) | **+0.42 points** |
| Permutation p, one-sided (9,999 draws, 4-week blocks) | **0.33** |
| Average and spread of the pretend deltas | -0.01 and 1.01 |
| HAC p, one-sided / HAC 95% interval | 0.36 / -1.83 to +2.67 |
| One-sided 95% upper bound (HAC) | 2.30 |
| Pre-registered MDE | 4.90 |
| **Verdict by the pre-stated rule** | **NOT SUPPORTED; effects as large as the pre-registered MDE are ruled out** |
| Qualifiers added by the rule | **design check failed (PL2, p = 0.038)**; Monday minus Wednesday moved the opposite way (PL1, p = 0.005), a warning about the design |

In plain words: after the swap, Tuesday's NET did not rise relative to Thursday's. The expiring index did not lead
the other more, on average, over the whole day. The test was also more sensitive than planned: the actual day-to-day
spread of NET within a cell is 3.18 points, half the 6.38 assumed from Nifty and Bank Nifty, which would put the MDE
at about 2.44 points (`verdict.json`; this number was not part of the decision rule).

## 5. Secondary tests (Holm-corrected as one family of three)

**Table 3 (REAL DATA, `tests.csv`).**

| Test | What | Estimate | p | HAC 95% interval | Holm p | Result |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | H1 in the last hour (14:15 to 15:14) | +2.51 | 0.0051 (permutation, one-sided) | +0.65 to +4.38 | **0.015** | supported |
| S2 | H2: total connectedness, either index's expiry day minus other days (weekday and Sensex-period effects) | -1.01 | 0.20 (HAC, two-sided) | -2.55 to +0.53 | 0.40 | not supported |
| S3 | H3: Sensex moves from Friday to Tuesday (January 2025); expected below zero | +0.81 | 0.60 (permutation, one-sided) | -2.38 to +4.00 | 0.60 | not supported (wrong sign) |

**S1, the last hour.** Cell means of last-hour NET (`cell_means.csv`, test S1): Tuesday before (Sensex expiry) -2.29
(median -0.79), Thursday before (Nifty expiry) +0.05, Tuesday after (Nifty expiry) -0.65, Thursday after (Sensex
expiry) -0.82. Tuesday rose by 1.64 and Thursday fell by 0.87, giving +2.51. A reading made after seeing the numbers
(not pre-stated): most of the effect comes from the Tuesday side, where the last hour of Sensex's own expiry days in
2025 leaned towards Sensex leading; the mean there (-2.29) is well below the median (-0.79), so a minority of days
carries much of it. After the swap, Sensex's Thursday expiries show no comparable lead (-0.82 against -0.65 on
Nifty's Tuesday expiries).

**S2, CPP's question on a new pair.** With weekday and Sensex-period effects in the regression, expiry days of either
index are 1.01 points less connected than other days, not significant (R squared 0.063). Without weekday effects
(description only, `robustness.csv` last row) the gap is -0.85 with HAC p = 0.00002, but there "expiry day" mostly
means "Tuesday or Thursday", so that version cannot separate expiry from weekday. The direction (less connected on
expiry days) matches what the connectedness study saw for Nifty and Bank Nifty (-2.75 points, p 0.18, post-hoc check E2
in [CONNECTEDNESS_GAMMA.md](CONNECTEDNESS_GAMMA.md)).

**S3, the January 2025 Sensex move.** Cell means (`cell_means.csv`, test S3): Friday before (Sensex expiry) +3.64
(7 days), Tuesday before (no expiry) +3.27 (6 days), Tuesday after (Sensex expiry) +3.60 (34 days), Friday after (no
expiry) +3.17 (31 days). Nothing moved much, and this test could only detect very large effects (MDE 9.67 points,
`mde.csv`).

## 6. Placebos and design checks (two-sided, not corrected)

**Table 4 (REAL DATA, `tests.csv`).**

| Placebo | What | Estimate | Permutation p | HAC p | HAC 95% interval | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| PL1 | Monday minus Wednesday around 1 Sep 2025, days on which no index expires | -4.96 | **0.0048** | 0.0025 | -8.18 to -1.75 | moved, opposite sign to delta: a warning |
| PL2 | Tuesday minus Thursday at a fake swap on 5 May 2025 (inside the before window) | +4.45 | **0.038** | 0.017 | +0.81 to +8.10 | **design check failed** |
| PL3 | Tuesday minus Thursday at a fake swap on 16 Feb 2026 (after window, before the auction) | +0.63 | 0.59 | 0.52 | -1.29 to +2.55 | passed |

* **PL2.** Between January to April 2025 and May to August 2025 nothing changed about who expires when, yet the
  Tuesday-minus-Thursday gap in NET moved by 4.45 points (`cell_means.csv`, test PL2: Thursday's Nifty expiry days
  fell from +6.10 to +2.41, Tuesday's Sensex days rose from +3.22 to +3.98). The design assumes this gap is stable
  when nothing changes. It was not, in the first half of the sample.
* **PL1.** Mondays fell from +4.51 to -1.38 and Wednesdays from +1.27 to +0.34 (`cell_means.csv`, test PL1). The
  pre-registration warned that the swap also changed which index expires the *next* day (Monday was the eve of a
  Sensex expiry before and of a Nifty expiry after). An eve-of-expiry lead by the expiring index would have pushed PL1
  up, in the H1 direction; it went down instead. So weekday gaps in NET also shift on days with no expiry at all.
* **PL3** passed: in the after window the gap was stable.

Together, PL1 and PL2 say that differences between weekdays in this measure can move by 4 to 5 points without any
change in expiry. That is the size of effect the study was built to find. The whole-day null result stands, but a
true effect of a few points could be hidden by such drift, and the positive last-hour result could be helped by it.

## 7. Robustness of the primary (pre-stated, not decisive, not corrected)

**Table 5 (REAL DATA, `robustness.csv`).** Each row changes one thing in the primary test.

| Check | Estimate | Permutation p | HAC p |
| --- | --- | --- | --- |
| Blocks of 1 week | +0.42 | 0.32 | 0.36 |
| Blocks of 2 weeks | +0.42 | 0.33 | 0.36 |
| Blocks of 8 weeks | +0.42 | 0.36 | 0.36 |
| Labels moved separately within each weekday | +0.42 | 0.41 | 0.36 |
| Closing-auction days left out (154 days) | +0.32 | 0.39 | 0.39 |
| Monthly expiry days of either index left out (139 days) | +0.09 | 0.47 | 0.47 |
| Input = intraday part of volatility only | **+4.71** | **0.015** | 0.0099 |
| VAR(10) chosen by AIC, horizon 30 | **+3.05** | **0.0077** | 0.0038 |
| One mcsGARCH fit over the whole sample | +0.37 | 0.32 | 0.36 |
| Week fixed effects (166 days in 83 weeks with both days) | +0.41 | 0.34 | |
| Ranks of NET instead of values (rank units) | +0.0002 | 0.50 | |
| Nifty to Sensex share alone (H1 implies a rise) | +1.94 | 0.0014 | 0.0002 |
| Sensex to Nifty share alone (H1 implies a fall) | +1.52 | 0.97 | 0.96 |
| Total connectedness (description) | +3.46 | 0.0004 (two-sided) | 0.00005 |
| Paper-style pooled VAR per cell | -0.64 | 0.63 | |

* The null result of the primary does not depend on the block length, the auction period, the monthly expiries, the
  separate volatility fits, week-to-week shocks, a few extreme days (ranks) or the per-day versus pooled VAR.
* Two settings that change the input or the VAR give a positive and nominally significant delta. These are the same
  sensitivity settings that weakened the Nifty and Bank Nifty result in the other direction
  ([CONNECTEDNESS_GAMMA.md](CONNECTEDNESS_GAMMA.md), section 5.5); such settings move connectedness numbers a lot.
* **The two halves of NET both rose.** On Nifty-expiry days, relative to Sensex-expiry days on the same weekday, the
  Nifty to Sensex share rose (+1.94) and so did the Sensex to Nifty share (+1.52). Their difference, NET, barely
  moved. In other words, total connectedness was higher on Nifty's expiry days than on Sensex's (+3.46; whole-day cell
  means: 93.11 against 92.05 before the swap, 93.76 against 91.35 after, `cell_means.csv`). This is description, not
  a pre-stated test, and it was not corrected for the 16 robustness rows.

## 8. What this means

1. **Over the whole day, the expiring index does not become the net transmitter of volatility between Nifty and
   Sensex**, at least not by the 4.9 points the study was built to detect, with the paper's settings.
2. **The last hour may be different.** The secondary last-hour test supports H1 after the Holm correction, but it leans
   on Sensex's Tuesday expiries of 2025, and the failed design check shows that weekday gaps in this measure drift
   by a similar amount on their own. It is a lead for a new pre-registered test on data after 2 November 2026, not a
   finding.
3. **Expiry seems to change how tightly the two indices are linked more than who leads.** Total connectedness was
   higher on Nifty's expiry days than on Sensex's in both periods (description only), while on expiry days of either
   index it was, if anything, slightly lower than on other days (S2, not significant).
4. **For the project:** the anchor paper's connectedness method, applied to a cleaner natural experiment than the
   paper had, gives no clear answer on direction. The day-to-day and weekday-to-weekday noise of directional measures
   is large relative to the effects of interest.

## 9. Known weaknesses

The pre-registration listed these before the results (section 14 there); the results add the first two.

1. **The design check failed (PL2)**, and the Monday/Wednesday placebo moved (PL1). The identifying assumption, a
   stable Tuesday-minus-Thursday gap without the swap, is in doubt.
2. **The result depends on settings**: the primary whole-day test finds nothing, while the last hour and two
   sensitivity settings point toward H1.
3. **Only the sum of the Nifty and Sensex expiry effects is identified**, not each one.
4. **The swap moved everything at once**: the weekly options, each exchange's monthly index and single-stock
   contracts, and which index expires the next day. "Nifty expiry day" means "NSE expiry day".
5. **NET at the one-minute scale is small for two near-identical indices** and may partly reflect how each exchange
   computes and publishes its index and how the provider stamps the bars.
6. **Data**: the Sensex bars are not validated against BSE's official values, and the Sensex calendar is reported, not
   verified (see [SENSEX_CONFOUND.md](SENSEX_CONFOUND.md)).
7. **Power**: the MDE rested on a variance borrowed from Nifty and Bank Nifty (the real spread was half of it); H3 has
   only 6 and 7 days before its event.
8. **The after window includes 13 weeks of the closing auction** (on both exchanges from 3 August 2026); leaving them
   out changes nothing (Table 5).
9. **Minute volatility is a model output** (mcsGARCH fitted in-sample per regime), as in the anchor paper.

## 10. Deviations from the pre-registration

**None.** The script, the tests and the pre-registration were run exactly as committed in 042d510; no analysis code
was changed, and the input files have the same SHA-256 digits as in phase 1. The reading of the last-hour cell means
in section 5 and the description of the two halves of NET in section 7 were written after seeing the results and are
labelled as such; they do not change any test or verdict.

## 11. How to reproduce

```
OMP_NUM_THREADS=1 python experiments/sensex_swap.py --coverage   # calendars and coverage only (phase 1)
OMP_NUM_THREADS=1 python experiments/sensex_swap.py              # the analysis, about 26 seconds
OMP_NUM_THREADS=1 python -m pytest -q tests/test_sensex_swap.py  # 18 tests on SYNTHETIC data
```

The analysis was run twice; every output file was identical except the run time (26.7 and 25.4 seconds in
`run_info.json`). Seeds: `numpy.random.default_rng([20261102, 92, key])`, keys in `run_info.json`.

| File (`results/sensex_swap/`) | Contents |
| --- | --- |
| `tests.csv` | primary, secondary and placebo tests, permutation and HAC p-values, Holm p |
| `cell_means.csv` | mean, SD and median of NET per cell, with the two shares and total connectedness |
| `robustness.csv` | the 16 robustness rows |
| `verdict.json` | the verdict by the pre-stated rule, its qualifiers, the observed spread |
| `h2_regression.csv` | all coefficients of the H2 regression and its description-only version |
| `day_values.csv` | one row per day: calendar flags and every NET, share and total connectedness value |
| `mcsgarch_fits.csv` | the volatility model per regime and index |
| `run_info.json` | pre-registration commit, settings, seeds, input hashes, run time |
| `calendar_days.csv`, `calendar_check.csv`, `coverage.csv`, `excluded_days.csv`, `mde.csv`, `mde_proxy.csv`, `run_info_coverage.json` | phase 1 (committed with the pre-registration) |

Logs: `results/sensex_swap_stdout.txt` (analysis) and `results/sensex_swap_coverage_stdout.txt` (phase 1).
