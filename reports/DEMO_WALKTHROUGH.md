# Demo walkthrough (Pandey prototype): about 6 minutes, every command run

Written by an AI agent (Claude), 3 November 2026, validation agent 3. Every command and output excerpt below was run in this session from the repository root with the project's Python environment (`python` below means that environment's interpreter) and `PYTHONPATH=src OMP_NUM_THREADS=1`. All numbers are REAL DATA (Nifty, committed forecast files) unless marked otherwise.

## Summary for a beginner (5 lines)

1. The demo takes one real trading day and shows, window by window, what the volatility really was and what two models had forecast at the start of each window.
2. Volatility here means how much the Nifty index moved up and down inside a 15-minute window (variance of one-minute returns, a bigger number means a wilder window).
3. The two models are mcsGARCH (our implementation of the paper's model) and the hybrid (mcsGARCH times a small learnt correction). Lower loss (QLIKE) means a better forecast.
4. Nothing is refitted during the demo: the script only reads committed result files, so it runs in under one second and works without internet.
5. If anything breaks, open the three prepared pages in `results/demo/` by double-clicking them: they are byte-identical to what the script writes (checked with `cmp`).

## Before the meeting (2 minutes, once)

| Check | Command | What I saw |
|---|---|---|
| Script runs | `python scripts/demo_forecast_day.py --pick` | three lines, one per day, in 0.35 s |
| Pages are current | rerun the three dates into a scratch folder and `cmp` with `results/demo/*.html` | all three identical |
| Pages open offline | opened in Chromium with the network switched off, at 1280x800 and 390x844 | no console errors, no external requests, no horizontal scroll |

Keep three things open: a terminal in the repository root, `results/demo/demo_nifty_2025-12-23.html` in a browser, and `docs/DEMO.md` (the longer guide).

## The flow

### Step 1 (30 s): say what we are testing

Say: "The paper fits an intraday volatility model, mcsGARCH, but never forecasts with it. I asked: can a small learnt correction on top of mcsGARCH forecast the next 15 and 30 minutes better, out of sample?"

Do not claim more: the gain is against our implementation of mcsGARCH and mostly comes from recent volatility and time of day, not from option-market information (study P106).

### Step 2 (45 s): show how the days were picked

```bash
python scripts/demo_forecast_day.py --pick
```

Expected output (excerpt, run today):

```
ordinary Tuesday expiry: 2025-12-23  QLIKE mcsGARCH 0.166, hybrid 0.155, gain +6.6%
    rule: Nifty expiry days on a Tuesday from 1 October 2025 on (51 days); the one with the median daily QLIKE gain ...
election-result crash: 2024-06-04  QLIKE mcsGARCH 3.343, hybrid 3.427, gain -2.5%
hybrid wins at the open: 2025-04-08  QLIKE mcsGARCH 0.185, hybrid 0.159, gain +13.7%
```

Say: "I did not cherry-pick: day 1 is the median-gain Tuesday expiry, day 2 was named in advance and is a day the hybrid does badly, day 3 is the median day where the hybrid wins at the open."

Note: here a positive "gain" means the hybrid is better. On the page itself the same quantity is written "hybrid vs mcsGARCH -6.6%" (negative means the hybrid's loss is lower). Say this aloud once or the signs will look contradictory.

### Step 3 (2 min): day 1, the ordinary Tuesday expiry

```bash
python scripts/demo_forecast_day.py --date 2025-12-23 --png        # optional, 0.5 s, writes the page and the deck chart
```

Expected (excerpt):

```
Next 15 minutes: QLIKE mcsGARCH 0.166 (570th highest of 792), hybrid 0.155 (578th), hybrid vs mcsGARCH -6.6% (350th best day for the hybrid)
Next 30 minutes: QLIKE mcsGARCH 0.129 (515th highest of 792), hybrid 0.121 (510th), hybrid vs mcsGARCH -6.3% (413th best day for the hybrid)
Last hour (14:30 to 15:30): QLIKE mcsGARCH 0.009 (642nd highest of 792), hybrid 0.023 (553rd), hybrid vs mcsGARCH +150.8% (697th best day for the hybrid)
Hybrid closer in 8 of 25 15-minute windows.
```

Open `results/demo/demo_nifty_2025-12-23.html`. Walk down the page:

* Three tiles: "mcsGARCH 0.166, hybrid 0.155, 6.6% lower". Say: "QLIKE is a forecast error score: 0 is perfect, bigger is worse. The hybrid's score is 6.6% lower today."
* Big chart (grey bars = what happened, blue dashed = mcsGARCH forecast, orange = hybrid forecast). Say: "At 09:15 mcsGARCH starts from yesterday's level and under-forecasts the busy open; the hybrid forecasts a busier open and wins that first window, loss 0.395 against 0.850. After that the two lines are almost on top of each other."
* Be ready for: "the hybrid is closer in only 8 of 25 windows". Say: "Yes. It wins big in a few windows (mainly the open) and loses a little in many, so it is 6.6% better over the day but is not better window by window."
* Be ready for the last-hour row: "+150.8%". That is the hybrid being worse on this day for the 14:30 to 15:30 window (one window, loss 0.009 against 0.023, both tiny). Say it plainly; it is one window on one day.
* Histogram "How unusual was this day": the hybrid beat mcsGARCH on 565 of 792 days. Median day 5.2% lower. This day is a typical one.
* P106 panel: "An AVERAGE over all 792 test days, not this day": recent realized volatility 34% of the gain, time of day 23%, option market inputs not significant. Say: "the gain is mostly a learnt correction of mcsGARCH, not extra information from options."

What the chart means for a beginner: grey bars are real market jumpiness in each quarter hour (high at the open and the close, low at lunch), lines are guesses made before each quarter hour, a better guess hugs the bars.

### Step 4 (1 min): day 2, the crash, shown as an honest failure

```bash
python scripts/demo_forecast_day.py --date 2024-06-04
```

Expected (excerpt): `Next 15 minutes: ... hybrid vs mcsGARCH +2.5%` and `Next 30 minutes: ... -23.9%`, and "Nifty ended the day 5.6% lower ... realized variance 70.8 times the median test day's".

Say: "On the election-result crash day the hybrid was slightly worse at 15 minutes and much better at 30. Both models had a terrible day, the 3rd and 2nd hardest of 792. I show it because a forecast should be judged on bad days too."

### Step 5 (45 s): day 3 and another index

```bash
python scripts/demo_forecast_day.py --date 2025-04-08
python scripts/demo_forecast_day.py --date 2025-06-10 --index BANKNIFTY
```

Expected: day 3 `hybrid vs mcsGARCH -13.7%` at 15 minutes. Bank Nifty page written, `QLIKE mcsGARCH 0.271 ... hybrid 0.242 ... -10.5%`. Say: "The same code on Bank Nifty, which was not used to design the hybrid."

### Step 6 (1 min): if the professor names a date

```bash
python scripts/demo_forecast_day.py --date 2025-09-19
```

Expected (run today): `Next 15 minutes: QLIKE mcsGARCH 0.341 ..., hybrid 0.286 ..., hybrid vs mcsGARCH -16.3%`, `Next 30 minutes: ... -28.6%`, and `Last hour: ... +104.2%` (the hybrid is worse in the last hour on this day; say so). Any test day from 2023-08-10 to 2026-11-02 works and writes `results/demo/demo_nifty_<date>.html`. Use a scratch folder with `--out /tmp/demo` if you do not want to touch `results/demo/`. If the date is not a test day, see the error table below. Say: "The model is walk-forward: every forecast on every page was made using only earlier days."

## Error states I tried (all exit code 2, nothing is written)

| Input | What the script printed | Say |
|---|---|---|
| `--date 2024-06-08` (a Saturday) | `2024-06-08 is not a test day for NIFTY (test days run from 2023-08-10 to 2026-11-02, 792 days).` plus the three nearest test days before and after | "Weekend, markets closed. Here are the nearest trading days." |
| `--date 2026-01-26` (Republic Day, holiday) | same message; nearest 2026-01-23 and 2026-01-27 | "Exchange holiday." |
| `--date 2023-08-01` (before the test period) | `Nearest test days before: none`, after: 2023-08-10 ... | "The first 150 trading days train the model, so the test starts 10 August 2023." |
| `--date 2025-13-45`, `--date abc`, `--date ""` | `Cannot read the date '...'; use YYYY-MM-DD.` | "Format error." |

I did not test an empty or corrupted forecast file: the data files are committed and read-only for the demo, so I judged it out of scope. If `results/pipeline_nse_v2/forecasts_next15.csv` is missing the script cannot run: use the fallback below.

## Fallback if live execution fails (laptop or Python broken)

1. Open the committed pages by double-click (no server, no internet): `results/demo/demo_nifty_2025-12-23.html`, `demo_nifty_2024-06-04.html`, `demo_nifty_2025-04-08.html`.
2. Use the deck chart `results/demo/deck_chart_2025-12-23.png` (1600 x 900) for the single-day picture.
3. Read the numbers from `docs/DEMO.md` (the same text, with the headline gains 5.8% at 15 and 7.6% at 30 minutes against our version, and 4.4% and 6.6% against mcsGARCH as the paper specifies it, P116).

## Known rough edges you should know about (not fatal)

* On a phone-width screen the main chart shrinks to about 6 px text. Use a laptop or the projector.
* The caption for a day whose volatility is near the median reads "about as busy than usual" (grammar slip in the script's caption template, seen on `--date 2026-07-15`).
* The terminal prints a pandas deprecation warning on some pandas versions; add `-W ignore` to hide it.
* The headline gain depends on the benchmark: always say "against our implementation of mcsGARCH" for 5.8% and 7.6%, and "4.4% and 6.6% against the paper's specification" when asked about the paper's own model.
