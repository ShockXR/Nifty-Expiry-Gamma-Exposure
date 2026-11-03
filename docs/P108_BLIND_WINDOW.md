# P108: which days does the paper's "before" sample really cover? (blind re-derivation)

Study P108, run by an AI agent (Claude) working blind on 3 November 2026.
All numbers below are REAL DATA unless marked SYNTHETIC. Every number comes from a file in
`results/p108_blind_window/` or from the paper (table and row given).

* Registered plan: `docs/prereg/P108_BLIND_WINDOW.md`, commit `6b289ca7b3f29803dfab0159377235eb7452761b`
  (committed 05:39:36 UTC, before any search on real data).
* Code: `experiments/p108_blind_window/search.py` (registered search), `experiments/p108_blind_window/posthoc.py`
  (POST HOC analyses, written after seeing the registered results). Tests (SYNTHETIC only):
  `tests/test_p108_blind_window.py`.
* Printed numbers typed from the paper: `experiments/p108_blind_window/printed_values.csv`.

## 1. The answer in plain words

**The paper says its "before" sample is November 2023 to February 2024. The printed numbers do not
fit that period at all. They fit a 64-trading-day window one year earlier, from late November 2022
to about 20 February 2023.**

* **Best window: 22 November 2022 to 20 February 2023 (64 trading days).**
  Runner-up, almost tied: **23 November 2022 to 21 February 2023** (the same window shifted by one day).
* **Return definition:** one-minute log returns of the close price, using 360 bars a day
  (09:15 to 15:14 IST), with the window's days joined into one series, so the overnight return
  from one day's 15:14 price to the next day's 09:15 price is included (except for the first day).
  This gives exactly 64 x 360 - 1 = 23,039 returns, the N printed in Table 1.
* **Confidence:**
  * that the sample lies in **late November 2022 to February 2023, not November 2023 to February 2024: very high**
    (Section 3);
  * that it is **exactly 22 November 2022 to 20 February 2023: moderate**. Under the registered rule
    this counts as a *weak* identification, because the window one day later ties on the main
    score. Everything that separates the two (the distance D, the Anderson-Darling and ADF values)
    leans towards 22 November to 20 February, but only slightly.
* **The stated window scores very badly.** November 2023 to February 2024 reproduces none of the 15
  printed numbers under any of the 204 return definitions, and ranks 85,013th of 132,126 windows under
  the winning definition. It contains one-minute returns far larger than the printed maximum and
  minimum (for example +1.35% for Nifty 50 at 09:15 on 4 December 2023, and -2.76% for Nifty Bank
  at 09:15 on 17 January 2024), so it cannot be the sample, whatever the details of the method.
* **The after window is as stated.** The same method finds that the paper's "after" numbers fit
  1 March to 31 May 2024 (POST HOC check in Section 5.6), matching the paper's text.

* **Added in phase 3, after the blind answer was pushed:** the project's earlier, independent search
  found the same year, length, return definition and explanations, but put the window about one
  week earlier (14 November 2022 to 10 February 2023). Section 8 explains the difference: the printed
  means favour this study's window, the printed higher moments favour the earlier one. The blind
  answer above is unchanged.

What this means for a beginner: the paper's before and after samples are not next to each other.
The before data seem to be from the winter of 2022 to 2023, roughly fifteen months before the
after data, not the four months just before the expiry change. Possibly the before data were taken
for the wrong year. Any "before versus after" difference in the paper therefore also contains
everything else that changed between early 2023 and mid 2024.

## 2. How the search worked (registered plan, short version)

1. **Printed numbers.** Table 1 (page 5) prints, for each index before the change: mean, standard
   deviation (SD), minimum, maximum, skewness, kurtosis and N of one-minute returns, plus their
   correlation, the Anderson-Darling (AD) statistic and the ADF statistic. Several are printed with
   9 or more digits, which makes them a fingerprint.
2. **Return definitions (204).** 17 sessions (the full 09:15 to 15:29 day, and every block of 360
   consecutive minutes inside it), 3 overnight treatments (joined series, within-day only,
   computed on the whole history then cut), log or simple returns, and 2 calendars (all days, or
   only full days).
3. **Windows.** Every run of 40 to 100 consecutive trading days between April 2017 and April 2026,
   about 132,000 windows per definition, about 27 million in all.
4. **Score.** E = how many of the 15 printed numbers (mean, SD, min, max, skewness, kurtosis and N for
   each index, plus the correlation) the window reproduces exactly at the printed number of decimals.
   Ties are broken by D, a distance (root mean square of 11 relative errors; 0 is a perfect fit).
5. **Stage 2.** The top windows are recomputed directly from prices, and the AD statistic is added (E2, out of 17).

## 3. Results of the registered search (REAL DATA)

### 3.1 The winner and its neighbours

Winning definition `0915-1514|concat|log|all` (09:15 to 15:14, joined series, log returns, all days;
the "full days" calendar gives the same result because there are no short sessions in this period).
Source: `results/p108_blind_window/best_by_definition.csv`, `top_candidates.csv`.

| | Printed (Table 1, Before) | 22 Nov 2022 to 20 Feb 2023 | 23 Nov 2022 to 21 Feb 2023 |
|---|---|---|---|
| N (each index) | 23,039 | 23,039 (match) | 23,039 (match) |
| Nifty mean | -0.000001 | -0.00000081 (match) | -0.00000115 (match) |
| Bank mean | -0.000002 | -0.0000018 (match) | -0.0000020 (match) |
| Nifty max | 0.006221108 | 0.006221108 (match) | 0.006221108 (match) |
| Bank max | 0.012023448 | 0.012023448 (match) | 0.012023448 (match) |
| Nifty min | -0.008584649 | -0.008533548 | -0.008533548 |
| Bank min | -0.017256735 | -0.017135337 | -0.017135337 |
| Nifty SD | 0.000360947 | 0.000361323 | 0.000362245 |
| Bank SD | 0.000522211 | 0.000525040 | 0.000527374 |
| Nifty skewness | -0.524960403 | -0.437 | -0.436 |
| Bank skewness | -1.357725832 | -1.130 | -1.122 |
| Nifty kurtosis | 64.74526352 | 65.0 | 64.3 |
| Bank kurtosis | 125.1117068 | 122.0 | 119.8 |
| Correlation | 0.854 | 0.8546 | 0.8535 |
| E (out of 15) | | **6** | **6** |
| D | | **0.0720** | 0.0746 |
| AD Nifty / Bank (printed 649.191 / 790.061) | | 637.2 / 790.2 | 619.3 / 762.1 |
| ADF Nifty / Bank, report only (printed -23.05000 / -20.69261) | | -23.04 / -20.84 | -22.56 / -20.87 |

(SD shown with N in the denominator, the variant that fitted best; skewness and kurtosis with the
best-fitting formula variant, see `top_candidates.csv`. ADF: constant, lag chosen by AIC, statsmodels.)

* The two printed **maxima are reproduced to all 9 printed decimals.** Searching the whole 2017 to 2026
  history under all 204 definitions, exactly one minute matches each: Nifty 50 at 09:20 on 30 January 2023
  (an intraday return), and Nifty Bank at 09:15 on 3 February 2023 (the overnight return from the
  15:14 price of 2 February). The Nifty Bank match only works if the day ends at 15:14, which is how
  the 09:15 to 15:14 session was singled out (`extreme_anchors.csv`).
* The two printed **minima are not reproduced exactly.** In our data both indices' smallest return
  in the window is the overnight return into Monday 30 January 2023 (from the 15:14 price of
  Friday 27 January), and it differs slightly from the printed one. Section 5.1 shows why.
* SD, skewness and kurtosis are close but not exact for any window, so our minute data and the
  paper's minute data are not identical in this period (Section 6).
* **Margin over the runner-up:** the window one day later ties on E (6 against 6) and has D only
  3.5% higher (0.0746 against 0.0720). Under the registered rule (Section 9 of the plan) this is a
  **weak identification**. The best clearly different window (sharing less than half its days),
  16 January to 21 April 2023, has E = 4 and D = 0.52.
* **Candidate set** (registered rule: E at least 5 and D at most twice the best, over all
  definitions): 13 windows, all with 63 to 66 trading days, all starting between 11 and 24 November
  2022 and ending between 9 and 22 February 2023, all under the 09:15 to 15:14 log-return definitions.

### 3.2 How rare is such a fit? (the null reference)

Under the winning definition, of 132,126 windows from 2017 to 2026 (`E_distribution_top_definitions.csv`):

| E | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| windows | 110,680 | 16,265 | 4,521 | 555 | 89 | 14 | 2 |

Only two windows reach E = 6, and they are the two above. All 16 windows with E of 5 or more start
between 11 November 2022 and 5 January 2023 and end between 9 February and 11 April 2023; none is
anywhere near November 2023 to February 2024.

### 3.3 Robustness to the return definition

From `best_by_definition.csv`:

* 131 of the 204 definitions put their best window inside October 2022 to March 2023; for the
  joined-series log-return definitions it is 32 of 34.
* Simple returns instead of log returns (same session): the same best window, 22 November 2022 to
  20 February 2023, but with E = 4, because the 9-decimal maxima match only for log returns.
* The single most frequent winner over all 204 definitions is 23 November 2022 to 21 February 2023
  (best under 44 definitions), followed by 18 November 2022 to 2 February 2023 (24) and
  22 November 2022 to 20 February 2023 (16). So the two leading windows swap places depending on
  the definition, but only the 09:15 to 15:14 log definitions reproduce the printed maxima and N.
* The full 375-minute session cannot give N = 23,039 and fits worse (best E = 3).

### 3.4 How the stated window scores

From `stated_windows.csv` (winning definition; "rank" is among 132,126 windows):

| Stated window variant | Days | E | D | Rank |
|---|---|---|---|---|
| 1 Nov 2023 to 29 Feb 2024 (the text's period) | 83 | 0 | 2.36 | 85,013 |
| first 64 trading days from 1 Nov 2023 (to 2 Feb 2024) | 64 | 2 | 3.33 | 3,819 |
| last 64 trading days to 29 Feb 2024 (from 30 Nov 2023) | 64 | 2 | 1.95 | 3,198 |

Over all 204 definitions the full stated window never reproduces any printed number (E = 0 at best,
D at least 1.13). Its ADF values (-101.8 for Nifty 50, -47.0 for Nifty Bank) are also far from the
printed -23.05 and -20.69, while the best window's (-23.04 and -20.84) are close. In the stated
window the Nifty 50 kurtosis is about 195 and the Nifty Bank one about 375, against 65 and 125 printed.

## 4. Other printed numbers (reported, not scored)

* **Wednesdays and Thursdays.** Table 5 prints 4,680 before observations on Wednesdays and on
  Thursdays, which is 13 days of 360 minutes. Both leading windows have 13 Wednesdays but only
  12 Thursdays (26 January 2023, Republic Day, was a Thursday holiday). The stated 83-day window has
  18 of each. So Table 5's Thursday count does not fit either; Table 5 is about trading volume and
  may have used other data.
* **Rogers and Satchell volatility (Table 6).** Our median per-minute value is about 0.00017 (Nifty 50)
  and 0.00026 (Nifty Bank) for the best window, against 0.00024 and 0.00034 printed; the stated window
  gives similar values (0.00018 and 0.00026). The paper's volatility measure must be defined differently,
  so it cannot help locate the window.

## 5. POST HOC analyses (after seeing the registered results; still blind to the earlier search)

These were not in the plan. They explain the mismatches; they do not change the blind answer above.
Code: `experiments/p108_blind_window/posthoc.py`; outputs `results/p108_blind_window/posthoc_*`.

### 5.1 The printed minima point to one different price on 27 January 2023

The printed minima can be rebuilt exactly from our prices if the last price of 27 January 2023 is the
**opening price of the 15:15 minute** instead of the closing price of the 15:14 minute:

* Nifty 50: ln(17461.95 / 17612.50) = -0.008584649, the printed minimum to all 9 decimals
  (our 15:14 close is 17611.60, which gives -0.008533548).
* Nifty Bank: ln(39674.85 / 40365.45) = -0.017256735, the printed minimum to all 9 decimals
  (our 15:14 close is 40360.55).

The same substitution works for both indices, whose minute files come from two different sources,
and nowhere else in 2017 to 2026 does "previous day's 15:15 open to 09:15 close" reproduce the printed
minima (`posthoc_ph1_minimum_pairs.csv`). So the paper's data almost certainly hold 17612.50 and
40365.45 as the 27 January 2023 day-end prices: a small difference between data vendors in one bar,
not a different window. Using the 15:15 open for every day instead breaks the Nifty Bank maximum
(on 2 February 2023 the paper's day-end price is our 15:14 close), so it is a one-day difference
(`posthoc_ph3_next_open_top10.csv`).

### 5.2 With that one price corrected, the same two windows lead

Replacing the two 27 January 15:14 closes and rescoring every window (`posthoc_ph2_patched_top30.csv`):
22 November 2022 to 20 February 2023 and 23 November 2022 to 21 February 2023 both reach E = 8
(N, mean, minimum and maximum for both indices) and nothing else does; D is 0.0628 against 0.0652.
SD, skewness and kurtosis still do not match exactly, so there are further small data differences.

### 5.3 The logic of "constraints" (why only two windows survive)

Among all windows with the patched prices, 16 reproduce N and all four extremes; all have 64 days,
start between 11 November and 2 December 2022 and end between 9 February and 2 March 2023. Only two
of the 16 also reproduce both printed means (the mean depends almost only on the first and last price
of the window): 22 November to 20 February and 23 November to 21 February
(`posthoc_ph7_feasible_windows_patched.csv`; the unpatched version gives the same 16 and the same two).

### 5.4 Does leaving out one day help? (no decisive answer)

Allowing any one interior day of a 65-day run to be missing (about 139,000 combinations) gives a
best E of 10 (21 November 2022 to 20 February 2023 without 28 December 2022, which also matches the
Nifty SD) but no combination reproduces the Nifty Bank SD, either skewness or either kurtosis, and
106 combinations reach E of 8 or more, so the extra freedom produces chance matches. This is not
evidence of a missing day (`posthoc_ph5_*`).

### 5.5 Stage 2 on the patched data

E2 is 8 for both leading windows (no exact AD match); AD and ADF again sit slightly closer to the
printed values for 22 November to 20 February (`posthoc_ph4_patched_stage2.csv`).

### 5.6 The after window confirms the method

With the same definition but dropping the two short Saturday sessions (2 March and 18 May 2024),
the best after window is **1 March to 31 May 2024**: 59 days, Nifty N = 21,239 exactly as printed
(59 x 360 - 1; our Nifty Bank file lacks 6 minutes in this period, so its N is 21,233), all four extremes reproduced at the 6 printed decimals, and E = 10 of 15, D = 0.007
(`posthoc_ph6_after_window_full_calendar.csv`). So the paper did use 09:15 to 15:14 joined log returns
and left out short special sessions, and its after window is the stated one. Only the before window
is off, by about a year.

## 6. Caveats

* **Different data.** Our minute bars come from two community datasets on GitHub, not from the
  paper's vendor. One price is shown to differ (Section 5.1) and the unmatched SD, skewness and kurtosis
  show there are more small differences. This is why the exact endpoints cannot be pinned down: the
  two leading windows differ only in the first and last day, and the scores that could separate them
  are the ones the data differences blur.
* **Endpoints.** 22 November 2022 to 20 February 2023 is the best guess; 23 November 2022 to
  21 February 2023 cannot be ruled out. Wider windows in the candidate set (start 11 to 24 November
  2022, end 9 to 22 February 2023) fit worse on the means.
* **Formula variants.** The best fits use SD with N in the denominator and the adjusted (Excel or
  pandas style) skewness; for kurtosis the best variant is raw (not excess) kurtosis. These choices
  are made per window by the score, so they are weakly identified.
* **Tables 5 and 6** (volume and volatility) do not fit any window cleanly (Section 4).
* **A coding detail.** Stage 2 picks the closest formula variant for each index separately, while
  stage 1 uses one variant for both indices; this changes only the third or fourth digit of the
  kurtosis shown and no conclusion.
* **Output format change.** After the first run, the code that writes the all-windows file was
  changed to a compact format (scores and a 15-character match mask for every window, plus full
  statistics for the top 5,000) to keep the repository small; the search was rerun and
  `best_by_definition.csv`, `top_candidates.csv` and `stated_windows.csv` were byte-identical to the
  first run.

## 7. Information barrier and files opened

The barrier was kept **by instruction, not enforced by the machine**: nothing stopped the agent from
reading the barred files; it simply did not. Until this document was committed and pushed, the agent
did not open, read, grep or list any barred file (the replication, re-key, event-study, real-data,
briefing, literature-review, referee-response, number-trace and test-ledger documents, the number
registry, the pitch, the README, the research strategy, the iteration log, the replication code and
its tests, anything under `results/` except this study's own output folder, and anything under
`/mnt/project-files/` except the paper), and did not import anything from the project package.

Things the agent saw that it should disclose:

* The `git reset --hard` used for setup printed the subject line of the branch's latest commit (about
  the number registry and trace counts). It says nothing about windows or dates.
* Listing `data/raw/` showed a folder named `extra_2022` (official daily files from 20 October 2022).
  Its name hints that 2022 mattered to someone, but the registered search covered 2017 to 2026
  uniformly and found the 2022 to 2023 window mechanically; the folder's daily file was used only
  after the search, to confirm that the minute files have no missing trading days in late 2022.

Files opened (read or loaded):

* `/mnt/project-files/Piyush_Pandey.pdf` (all 16 pages as images; text layer of pages 5, 12 and 13).
* `data/raw/nifty_1min.csv`, `data/raw/banknifty_1min.csv` (loaded in full), `data/raw/minute_bars_source.json`,
  `data/raw/index_daily.csv` (first lines, then the trading-day list for October 2022 to March 2023),
  `data/raw/extra_2022/index_daily.csv` (trading-day list, after the search).
* `scripts/fetch_minute_bars.py` (header, lines 1 to 80), `.gitignore`, `pyproject.toml`.
* Directory listings (names only): the repository root, `data/`, `data/raw/`, `data/raw/extra_2022/`,
  `data/raw/nse_full/`, `data/raw/finnifty/`.
* This study's own files: `docs/prereg/P108_BLIND_WINDOW.md`, `docs/P108_BLIND_WINDOW.md`,
  everything in `experiments/p108_blind_window/` and `results/p108_blind_window/`,
  `tests/test_p108_blind_window.py`, and scratch scripts in the agent's private scratch folder.
* **Only after the phase 2 push (phase 3):** `docs/REPLICATION_NSE.md` (a keyword search of the whole
  file for "before", "window" and "fingerprint", then Sections 2 to 4, lines 64 to 127, in full) and
  `results/replication_nse/fingerprint_search.csv`. No other barred file was opened, and for the
  final test run the agent ran the whole test suite, which executes the barred test files without
  the agent reading them.

## 8. Comparison with the earlier search (phase 3, written after the blind answer was pushed)

Everything above was committed and pushed (commit `d904c8c58f9d68816a82d656f13be817df955f6e`,
06:00 UTC) before the agent opened anything behind the barrier. Only then did it read
`docs/REPLICATION_NSE.md` (its summary and Sections 3 and 4, on the before window) and
`results/replication_nse/fingerprint_search.csv`. The blind answer in Sections 1 to 5 is unchanged.

### 8.1 What the earlier search found

The earlier search scored the 348 windows of exactly 64 trading days between July 2022 and February
2024, under the 09:15 to 15:14 rule (which it chose from the after window), on how closely the SD,
minimum, maximum and excess kurtosis match Table 1. Its best window is **14 November 2022 to
10 February 2023** (score 0.045), and it notes that neighbours a few days earlier or later score
almost as well.

### 8.2 Where the two searches agree

On almost everything, reached independently:

* the before sample is **64 trading days in November 2022 to February 2023**, not November 2023 to
  February 2024 (the earlier search ranks the best window starting in November 2023 or later 217th of 348;
  here the stated window scores E = 0 under all 204 definitions);
* the returns are **09:15 to 15:14 one-minute log returns, run straight through from one day to the next**;
* both printed maxima are the same two minutes (30 January 2023 09:20 for Nifty 50, 3 February 2023
  09:15 for Nifty Bank), matched to every printed decimal;
* both printed minima are the 30 January 2023 opening gap, and both searches trace the small
  mismatch to the same cause: the paper's day-end prices on 27 January 2023 equal our 15:15 opening
  prices (17,612.50 and 40,365.45), a one-bar difference between data feeds;
* the after window is 1 March to 31 May 2024 with 21,239 Nifty returns (and 21,233 for Nifty Bank in
  our files);
* Table 5's weekday counts do not fit (13 Wednesdays but 12 Thursdays in the window).

Where the two searches computed the same statistic on the same window they agree to the digits
shown (for example, for 14 November 2022 to 10 February 2023, Nifty 50 SD 0.0003578019 in both, and
Anderson-Darling 660.5 and 822.5 there against 660.46 and 822.39 here).

### 8.3 Where they disagree: the exact endpoints, by about one week

| Window (64 trading days) | Earlier search rank (of 348) | This study, registered rank (of 132,126) | E | Both printed means reproduced? |
|---|---|---|---|---|
| 14 Nov 2022 to 10 Feb 2023 (earlier answer) | 1 | 9 | 5 | No: Nifty Bank mean is -0.00000055, which prints as -0.000001, not -0.000002 |
| 22 Nov 2022 to 20 Feb 2023 (this answer) | 8 | 1 | 6 | Yes |
| 23 Nov 2022 to 21 Feb 2023 | 9 | 2 | 6 | Yes |

Both answers lie inside each other's near-ties, and inside this study's registered candidate set
(start 11 to 24 November 2022, end 9 to 22 February 2023). The searches differ by six trading days
because they listen to different printed numbers:

* **This study** counts exact matches first, and the printed **means** are among them. A mean of
  one-minute returns joined across days depends almost only on the window's first and last prices
  (it is the total log change divided by N). The printed means (-0.000001 for Nifty 50, -0.000002 for
  Nifty Bank) need Nifty 50 to fall between about 1.2% and 3.5% over the window and Nifty Bank between
  about 3.5% and 5.8%. Among the 16 windows of 64 days that reproduce N and the four extremes, only
  the windows starting 22 and 23 November do both. For 14 November to 10 February, Nifty Bank falls
  only about 1.3%; it would need to fall about 2.2 percentage points more to print as -0.000002.
* **The earlier search** scores SD, minimum, maximum and excess kurtosis, and leaves out the means.
  These numbers are printed very precisely, but they are driven by a handful of large returns, and
  the two data feeds are known to differ (Section 5.1); for example our Nifty Bank skewness is at
  least 14% off for every one of the 16 feasible windows (14% to 30%), so some large Nifty Bank
  returns must differ too.

**POST HOC check (phase 3)** (`experiments/p108_blind_window/phase3_compare.py`,
`results/p108_blind_window/phase3_feasible_windows_fixed_formulas.csv`). The registered search let each
window choose its kurtosis formula, and it chose raw kurtosis (excess plus 3) for 22 November to
20 February. The after window, and the earlier search, both indicate that the paper reports
**excess** kurtosis (after window: our excess kurtosis 75.92 against printed 75.88; raw would be
about 78.9). This was not used in the registered plan, which is a weakness of this study's score.
Rescoring the 16 feasible windows with the formulas fixed (SD with N in the denominator, adjusted
skewness, adjusted excess kurtosis) and ranking them by the distance alone:

| Window (with the 27 Jan 2023 price corrected) | Distance, fixed formulas | Rank of 16 | Both means? | Nifty kurtosis error | Nifty Bank kurtosis error |
|---|---|---|---|---|---|
| 16 Nov 2022 to 14 Feb 2023 | 0.0440 | 1 | No | -1.6% | -0.2% |
| 14 Nov 2022 to 10 Feb 2023 | 0.0568 | 7 | No | -0.9% | +0.8% |
| 22 Nov 2022 to 20 Feb 2023 | 0.0647 | 9 | Yes | -3.8% | -3.9% |
| 23 Nov 2022 to 21 Feb 2023 | 0.0684 | 11 | Yes | -5.0% | -5.8% |

(Kurtosis errors are log ratios, roughly percentages.) So with excess kurtosis, the precise
higher moments favour windows starting 14 to 17 November, as the earlier search found, while the
means favour 22 or 23 November. With the registered ranking (exact matches first, then distance),
fixing the kurtosis formula does not change the order of the top two, because neither reproduces
the kurtosis exactly and the means still decide.

### 8.4 Conclusion of the comparison

* **The two independent searches agree** on the substance: the paper's before sample is a 64-day
  window in late 2022 to early 2023, built from 09:15 to 15:14 log returns joined across days, and
  the stated November 2023 to February 2024 window is ruled out.
* **They disagree on the exact days, by about one week.** A fair joint statement is: the window
  starts between 14 and 23 November 2022 and ends between 10 and 21 February 2023 (64 trading days).
  Within that range, the means point to 22 November 2022 to 20 February 2023 (this study's blind
  answer, kept unchanged) and the precise higher moments point to mid-November (the earlier answer).
* Which is right depends on which printed numbers are more trustworthy. The means are printed with
  one significant digit but cannot be moved by small feed differences; the higher moments are printed
  with many digits but are moved by them. In the agent's judgement the means are the safer guide,
  unless the paper's Mean row was computed on a different series. Only the authors' data or code can
  settle it.
* Practical consequence: for any re-analysis, results should be checked on both windows (they share
  58 of their 64 days), as the earlier study already did with shifted windows.

## 9. How to reproduce

```
ln -s <path to raw data> data/raw
OMP_NUM_THREADS=1 python experiments/p108_blind_window/search.py --raw data/raw --out results/p108_blind_window --plan-commit 6b289ca7b3f29803dfab0159377235eb7452761b
OMP_NUM_THREADS=1 python experiments/p108_blind_window/posthoc.py --raw data/raw --out results/p108_blind_window
OMP_NUM_THREADS=1 python experiments/p108_blind_window/phase3_compare.py --raw data/raw --out results/p108_blind_window
PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 python -m pytest -q tests/test_p108_blind_window.py
```

Run details (inputs' SHA-256, versions, run time of about three minutes) are in
`results/p108_blind_window/run_info.json` and `posthoc_run_info.json`.
