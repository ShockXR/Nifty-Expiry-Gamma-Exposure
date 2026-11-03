# P111: meeting demo of the forecasting pipeline, one day at a time

Prepared by an AI agent (Claude) for Shreyan Thanki's meeting with Prof. Piyush Pandey, 3 November 2026.
**Everything shown is REAL DATA** (Nifty, NSE data as in the headline run). Every number below is printed by
`scripts/demo_forecast_day.py` from committed files, or read from the committed file named next to it.

## 1. What the demo shows, in one paragraph

The anchor paper fits mcsGARCH, an intraday volatility model, but never forecasts with it. Our hybrid takes
mcsGARCH's forecast and multiplies it by a correction learnt by LightGBM. Over all 792 test days it forecasts
the next 15 minutes 5.8% better and the next 30 minutes 7.6% better in QLIKE loss. Study P106 found where that
gain comes from: recent realized volatility (about a third) and the time of day (about a quarter), not the
option market. The demo page makes this concrete for one trading day: what really happened in each
15-minute window, what each model forecast at the start of that window, which one was closer, and how that
day compares with all the others.

## 2. Before the meeting

Run from the repository root. Each command takes a few seconds; nothing is refitted, and only committed files
are read.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/demo_forecast_day.py --pick
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/demo_forecast_day.py --date 2025-12-23 --png
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/demo_forecast_day.py --date 2024-06-04
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/demo_forecast_day.py --date 2025-04-08
```

The pages are already in `results/demo/`, so nothing has to run during the meeting:

| File | What it is |
|---|---|
| `results/demo/demo_nifty_2025-12-23.html` | Day 1, an ordinary Tuesday expiry |
| `results/demo/demo_nifty_2024-06-04.html` | Day 2, the election-result crash |
| `results/demo/demo_nifty_2025-04-08.html` | Day 3, a day when the hybrid wins at the open |
| `results/demo/deck_chart_2025-12-23.png` | Day 1's session chart for the slide deck, 1600 x 900 pixels |

Open a page by double-clicking it (or `xdg-open`, `open` or `start` followed by the file name). It is a single
file with no internet connection needed. Hovering over a window shows its numbers; the table under
"Every window of the day" lists them all.

**If the professor asks for another date,** run the same command with that date. If it is not a test day (a
holiday, a weekend, or before 10 August 2023), the script prints the nearest test days and stops without
writing anything:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/demo_forecast_day.py --date 2024-06-08
```

## 3. How to read one page (say this once, on the first day)

* **The heading and caption** give the date and one plain sentence on what kind of day it was. The size of the
  move comes from `daily_market.csv` (the last one-minute price of the day against the day before); the
  "times the median test day" figure compares the day's realized variance with the median test day's. Event
  notes such as "general election results day" are public news, not computed.
* **The three tiles** give each model's QLIKE for the day, the average over the day's 25 forecast windows.
  QLIKE is 0 for a perfect forecast and grows as the forecast misses; forecasting too low is punished more than
  forecasting too high. The rank says where the day sits among all 792 test days (1st highest = the hardest
  day for that model).
* **The big chart.** Grey bars are what happened: the realized variance in each 15-minute window. The blue
  dashed line is mcsGARCH's forecast and the orange solid line is the hybrid's, each made at the start of the
  window, before the window was seen. The coloured strip under the chart says which forecast had the lower
  QLIKE in each window (M = mcsGARCH, H = hybrid). Colours are paired with line styles, marker shapes and
  letters, so the chart reads without colour.
* **"How unusual was this day?"** is a histogram of the hybrid's daily gain over mcsGARCH on all 792 test days;
  the black bar holds the chosen day. The hybrid beat mcsGARCH on 565 of the 792 days, and on the median day its
  QLIKE was 5.2% lower.
* **The P106 panel** is the same on every page. It is an **average over all 792 test days, not that day**: P106
  refitted the hybrid without one group of inputs at a time and measured how much of the gain was lost.
* **The horizon table** repeats the day's QLIKE for the 30-minute and last-hour forecasts.

## 4. How the three days were chosen

Each day comes from a rule fixed before looking at that day's page. `--pick` prints the result:

```
ordinary Tuesday expiry: 2025-12-23  QLIKE mcsGARCH 0.166, hybrid 0.155, gain +6.6%
    rule: Nifty expiry days on a Tuesday from 1 October 2025 on (51 days); the one with the median daily QLIKE gain of the hybrid over mcsGARCH
election-result crash: 2024-06-04  QLIKE mcsGARCH 3.343, hybrid 3.427, gain -2.5%
    rule: named in advance; by the hybrid's daily QLIKE minus mcsGARCH's it is the 6th worst day for the hybrid (the worst is 2026-07-15)
hybrid wins at the open: 2025-04-08  QLIKE mcsGARCH 0.185, hybrid 0.159, gain +13.7%
    rule: non-expiry days on which the hybrid's 09:15 forecast had the lower QLIKE (406 days); the one with the median 09:15 QLIKE gain
```

In words:

1. **Ordinary Tuesday expiry.** Nifty's weekly expiry moved to Tuesday on 1 September 2025. Among the 51 Nifty
   expiry days that fall on a Tuesday from 1 October 2025 to the end of the test period, take the one whose
   daily gain (mcsGARCH's QLIKE minus the hybrid's, as a % of mcsGARCH's) is the median. October onwards keeps
   the choice strictly "after September 2025". With an even count the rule would take the lower of the two
   middle days; 51 is odd, so the median is a single day.
2. **The 4 June 2024 crash.** It was named in the brief before any page was made, so it was not chosen by its
   result. For reference, by the size of the hybrid's daily loss above mcsGARCH's it is the 6th worst of the
   792 days for the hybrid; the worst is 15 July 2026 (run `--date 2026-07-15` to show it).
3. **A typical day when the hybrid wins at the open.** Among the 406 non-expiry test days on which the hybrid's
   09:15 forecast had the lower QLIKE (days already picked are skipped), take the day whose 09:15 gain
   (mcsGARCH's QLIKE minus the hybrid's in that window) is the median. "Typical" is about the size of the win
   at the open; the day it picks, 8 April 2025, happens to follow the 7 April tariff sell-off and is busier
   than usual.

## 5. Day 1: an ordinary Tuesday expiry (23 December 2025)

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/demo_forecast_day.py --date 2025-12-23 --png
```

Page: `results/demo/demo_nifty_2025-12-23.html`. Deck chart: `results/demo/deck_chart_2025-12-23.png`.

**What the page shows** (printed by the script):

| Horizon | mcsGARCH QLIKE (rank of 792, 1st = hardest) | Hybrid QLIKE (rank) | Hybrid vs mcsGARCH (rank, 1st = hybrid's best day) |
|---|---|---|---|
| Next 15 minutes | 0.166 (570th) | 0.155 (578th) | -6.6% (350th) |
| Next 30 minutes | 0.129 (515th) | 0.121 (510th) | -6.3% (413th) |
| Last hour | 0.009 (642nd) | 0.023 (553rd) | +150.8% (697th) |

Caption: a Nifty weekly expiry day; Nifty ended almost unchanged, and the session was calmer than usual
(realized variance 0.5 times the median test day's).

**What to say:**

* "This is the median Tuesday expiry since October 2025, picked by rule. On this day the hybrid's loss is 6.6%
  lower than mcsGARCH's, close to its 5.8% average over all days."
* Point at the first window. "At 09:15 the market moved a lot: realized variance 3.03 (in units of 10^-6).
  mcsGARCH forecast 1.04. The hybrid's LightGBM correction raised that by a factor of 1.35, to 1.40. QLIKE for
  that one window: 0.850 for mcsGARCH, 0.395 for the hybrid."
* Point at the strip. "The hybrid was closer in only 8 of the 25 windows, yet it wins the day. Most windows
  are near-ties; the open is where the two models differ most, and QLIKE punishes the forecast that was too
  low. Over all 792 test days, the hybrid's QLIKE in the 09:15 window is 13.3% below mcsGARCH's
  (`results/pipeline_nse_v2/gain_by_anchor_next15.csv`)."
* Point at the P106 panel. "This is why: on average, about a third of the hybrid's gain comes from recent
  realized volatility and about a quarter from the time of day. The open is exactly where a time-of-day
  correction matters."
* If asked about the last hour: "Both losses that day are tiny (0.009 and 0.023) and there is only one
  last-hour forecast per day, so a single day's percentage there is noisy. Over all days the hybrid is 6.4%
  better in the last hour."

## 6. Day 2: the election-result crash (4 June 2024)

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/demo_forecast_day.py --date 2024-06-04
```

Page: `results/demo/demo_nifty_2024-06-04.html`.

| Horizon | mcsGARCH QLIKE (rank of 792) | Hybrid QLIKE (rank) | Hybrid vs mcsGARCH (rank) |
|---|---|---|---|
| Next 15 minutes | 3.343 (3rd) | 3.427 (2nd) | +2.5% (633rd) |
| Next 30 minutes | 3.072 (2nd) | 2.337 (3rd) | -23.9% (58th) |
| Last hour | 0.008 (648th) | 0.010 (640th) | +19.2% (543rd) |

Caption: general election results day; the count showed the ruling party short of a majority on its own, and
the market crashed. Nifty ended the day 5.6% lower (last one-minute price against the day before), and the
session's realized variance was 70.8 times the median test day's.

**What to say:**

* "This is the day we promised to show, whatever it looked like. It is one of the three hardest days of the
  whole test period for both models, and at 15 minutes the hybrid lost: its loss is 2.5% higher."
* Point at the first bar. "In the first window the realized variance was 486. Both models forecast about 6
  (5.99 and 5.73), because nothing in past prices said the count would surprise the market. That one window has
  a QLIKE of 75.7 for mcsGARCH and 79.4 for the hybrid, and it alone makes up most of the day's average of 3.34
  and 3.43."
* Point at the strip. "After the open, the hybrid was closer in 18 of the 25 windows: once the shock was in
  recent realized volatility, the correction used it. At 30 minutes the hybrid was 23.9% better on this day."
* The lesson: "A forecasting model built from past prices cannot see news coming. The hybrid's edge is that it
  reacts faster to what has already happened, which is the realized-volatility group in the P106 panel."

## 7. Day 3: a typical day when the hybrid wins at the open (8 April 2025)

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/demo_forecast_day.py --date 2025-04-08
```

Page: `results/demo/demo_nifty_2025-04-08.html`.

| Horizon | mcsGARCH QLIKE (rank of 792) | Hybrid QLIKE (rank) | Hybrid vs mcsGARCH (rank) |
|---|---|---|---|
| Next 15 minutes | 0.185 (495th) | 0.159 (559th) | -13.7% (148th) |
| Next 30 minutes | 0.179 (309th) | 0.148 (379th) | -17.6% (145th) |
| Last hour | 0.065 (424th) | 0.079 (376th) | +22.0% (554th) |

Caption: the trading day after the global tariff sell-off of 7 April 2025; Nifty ended the day 1.4% higher,
and the session's realized variance was 6.9 times the median test day's.

**What to say:**

* "This day was picked as the median among the 406 ordinary days on which the hybrid won the 09:15 window."
* Point at the first window. "After the previous day's sell-off, mcsGARCH expected a very volatile open: it
  forecast 57.6. What happened was 26.2. The hybrid's correction cut mcsGARCH's forecast by a factor of 0.81,
  to 46.6, still too high but closer: QLIKE 0.242 for mcsGARCH and 0.138 for the hybrid."
* "So the correction is not always 'raise the forecast'. On 23 December it raised the open's forecast; here it
  lowered it. It learns from recent realized volatility and the time of day when mcsGARCH tends to be too high
  or too low."
* "Over the whole day the hybrid was closer in 14 of 25 windows and its loss was 13.7% lower."

## 8. Closing: the P106 panel

The panel on every page shows, for the 15-minute forecasts, the share of the hybrid's gain lost when one group of
inputs is removed (`results/p106_gain_decomposition/shares.csv`), with the 30-minute share at the right:

| Group | 15 min | 30 min |
|---|---|---|
| Recent realized volatility (12 inputs) | 34% | 38% |
| Time of day (4 inputs) | 23% | 25% |
| Option market + VIX together (17 inputs) | 10% | -1% |
| Option prices (16 inputs) | 2% | -4% |
| India VIX (1 input) | -3% | -4% |
| Calendar and expiry (8 inputs) | -7% | -8% |

What to say:

* "Only realized volatility and time of day pass P106's registered test, at both horizons (dark bars). The
  option-market inputs add nothing we can detect."
* "The five separate groups add up to 49% at 15 minutes. The rest is shared: remove one group and others partly
  stand in for it."
* "A hybrid that sees only what mcsGARCH itself sees (past one-minute returns, the time of day and the
  expiry-day flag) is 6.1% better than mcsGARCH at 15 minutes, against 5.8% for the full hybrid
  (`results/p106_gain_decomposition/contributions.csv`). So the gain is a better use of mcsGARCH's own kind of
  information, not new information from the option market."
* Remind: "This panel is an average over all 792 test days. It does not say what drove any single day."

## 9. Things to be honest about

* **One day proves nothing.** The pages illustrate the average result; the evidence is the 792-day comparison
  and the P106 tests. The histogram on each page shows how widely single days vary: on the hybrid's worst day
  in percentage terms its QLIKE was 52% above mcsGARCH's, and on its best day 36% below (computed from
  `results/pipeline_nse_v2/forecasts_next15.csv`).
* **"Close" is the last one-minute price**, not NSE's official closing price, which is computed differently.
  The caption says so.
* **The 5.8% headline is a ratio of averages** (average QLIKE over all days). The median day's gain, 5.2%, is a
  different summary of the same comparison.
* **Ranks** count from the top: 1st highest loss = the hardest day; for the gain, 1st = the hybrid's best day.
* **The hybrid shown is `hybrid_base`**, the model behind the headline numbers, from the walk-forward run in
  `results/pipeline_nse_v2/` (refitted every 21 trading days on past data only).

## 10. Files

* Script: `scripts/demo_forecast_day.py` (options: `--date`, `--pick`, `--index NIFTY|BANKNIFTY|FINNIFTY`,
  `--out`, `--png`, `--label`, `--note`, `--results`, `--p106`).
* Tests (SYNTHETIC data only): `tests/test_demo_forecast_day.py`.
* Pages and deck chart: `results/demo/`.
* Inputs read: `results/pipeline_nse_v2/forecasts_next15.csv`, `forecasts_next30.csv`,
  `forecasts_last_hour.csv`, `daily_market.csv`; `results/p106_gain_decomposition/shares.csv`,
  `contributions.csv`, `run_info.json`, `verdict.csv`.
