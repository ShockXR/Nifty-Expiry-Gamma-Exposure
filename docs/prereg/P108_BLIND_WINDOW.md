# P108 registered plan: a blind search for the paper's "before" window

Written by an AI agent (Claude) on 3 November 2026, before any search was run on real data.
Status: registered plan (phase 1). The answer will be written to `docs/P108_BLIND_WINDOW.md` (phase 2).

## 1. The question

The anchor paper (Chhimwal, Pandey and Pandey 2025, Review of Derivatives Research, "CPP")
compares one-minute Nifty 50 and Nifty Bank returns in a "before" window and an "after"
window around the change of Nifty Bank's expiry day. Its text says the before window is
"November 2023 to Feb 2024" (Section 2.1, page 5), with no day of the month.
Table 1 (page 5) prints descriptive statistics of the before returns, with many digits.

**Question:** using only the paper's printed numbers and the raw minute data, which calendar
window of trading days does the before sample actually cover, and how sure can we be?
The stated dates are treated as a claim to test, not as a fact.

## 2. Information barrier (kept by instruction, not enforced by the machine)

Another part of the project already ran a search like this one. To keep this study
independent, the agent was told not to open, read, grep or list a fixed set of files
(the replication and re-key documents, the event-study and real-data result documents,
the briefing, the literature review, the referee responses, the number trace, the test
ledger, the number registry, the pitch, the README, the research strategy, the iteration
log, the replication code and its tests, everything under `results/`, and everything under
`/mnt/project-files/` except the paper's PDF), and not to read other commits' messages or
diffs, until the blind answer is committed and pushed. Nothing on the machine enforces this;
the agent keeps it by following the instruction. The phase 2 document will list every file
the agent opened.

## 3. What was looked at before writing this plan

* The paper's PDF (all 16 pages), read with a PDF reader, and its text layer for Tables 1, 5 and 6
  (to check digits).
* The format of the raw minute files `data/raw/nifty_1min.csv` and `data/raw/banknifty_1min.csv`
  (columns timestamp, open, high, low, close; IST times) and `data/raw/minute_bars_source.json`,
  plus the header of `scripts/fetch_minute_bars.py`. A whole-file count of bars per day
  (no returns, no statistics, no window) showed that nearly every day has 375 bars,
  09:15 to 15:29, and that a few days are special short sessions (for example evening
  sessions of about 60 bars, and Saturday sessions of about 105 bars).
* Nothing else in the repository was opened. No return was computed on real data before this plan.

## 4. What the printed numbers already say (arithmetic on printed numbers only)

All values are transcribed in `experiments/p108_blind_window/printed_values.csv` with
page, table, row and column.

* Table 1 gives N = 23,039 returns for each index before, and 21,239 after.
  23,039 = 64 x 360 - 1 and 21,239 = 59 x 360 - 1. Table 5 gives 4,680 observations on
  Wednesdays and on Thursdays before (13 x 360), and 4,320 on Wednesdays after (12 x 360).
  This suggests the paper's data had **360 one-minute bars a day** (not the 375 of the full
  09:15 to 15:29 session), and that returns were taken on one joined price series
  (64 days x 360 prices give 23,039 differences, so the first day's overnight return is
  missing and the other 63 overnight returns are included). This is a hypothesis to test,
  not an assumption: the search also tries the full 375-bar session and other treatments.
* If the before window had 64 trading days, it is shorter than November 2023 to February 2024
  (about 80 trading days). This is one reason to test the stated dates.
* Section 3.1 says the model uses "the log return of the intraday series". It does not say
  which session hours were used, whether overnight returns were kept, or which formulas
  were used for skewness and kurtosis (and whether kurtosis is excess kurtosis).
* Ambiguities noted: Table 1 prints the after Nifty Bank Anderson-Darling value as
  "466,153" (a comma where a decimal point is expected); the after N differs between
  Table 1 (21,239), Table 5 (21,463) and Table 6 (21,462). These affect only the after window.

## 5. Data (REAL DATA)

* `data/raw/nifty_1min.csv` (Nifty 50) and `data/raw/banknifty_1min.csv` (Nifty Bank),
  community one-minute bars (see `data/raw/minute_bars_source.json`). Close prices are used.
* Only bars from 09:15 to 15:29 are kept (evening special sessions fall outside and drop out).
* The trading-day calendar is the set of dates on which both files have at least one bar
  in the session being used. Days present in only one file are left out.

## 6. Return definitions tried (204 in all)

Every combination of:

1. **Session** (17): the full session 09:15 to 15:29 (375 bars), and every block of 360
   consecutive bars inside it (start 09:15, 09:16, ..., 09:30; for example 09:15 to 15:14
   or 09:30 to 15:29). Reason: the printed counts point to 360 bars a day, but which 360
   is unknown, so all 16 are tried; the full session is the natural default.
2. **Overnight treatment** (3):
   * `concat`: join the window's prices into one series and take differences
     (overnight returns included, except the first day's; N = bars - 1). This is what the
     printed N suggests.
   * `intraday`: differences within each day only (no overnight returns).
   * `concat_pre`: returns computed on the whole history first and then cut to the window
     (all overnight returns included, also the first day's).
3. **Return type** (2): log, ln(P_t / P_{t-1}) (what Section 3.1 says), and simple, P_t / P_{t-1} - 1.
4. **Calendar** (2): `all` (every common day, including short special sessions) and `full`
   (only days on which both indices have at least 90% of the session's bars). Reason:
   a researcher may or may not have dropped short days.

A missing minute inside a day is skipped (the return then spans two minutes).

## 7. Candidate windows

Every window of consecutive trading days (in the calendar of the definition) with a length
of **40 to 100 trading days**, starting anywhere in the common data, which runs from
April 2017 to April 2026. This is far wider than "a year either side" of the stated period:
more than six years before and about two years after November 2023 to February 2024.
Reason for the lengths: 360 or 375 bars a day and N = 23,039 imply 61 to 64 days; the
stated four months are about 80 trading days and the introduction mentions "90 days";
40 to 100 covers all of these with room to spare. Searching all years also gives a
built-in "null" reference: how often a window far from the stated period fits by chance.
About 137,000 windows per definition, about 28 million in all.

## 8. The score

For each window, compute for each index: N (number of returns), mean, standard deviation,
minimum, maximum, skewness and kurtosis, and the Pearson correlation between the two indices'
returns (on minutes where both have a return).

Formula variants, because the paper does not say which it used (all windows get the same
choice): standard deviation with N - 1 or N in the denominator; skewness g1 (population),
G1 (adjusted, as in Excel and pandas) or b1 (third moment over the sample SD cubed);
kurtosis g2, G2 or b2 in the same three ways, each either as excess kurtosis or raw (plus 3).
For each window and each statistic, the variant is chosen that gives the most exact matches
over the two indices, then the smallest error; the same variant is used for both indices.

* **E (primary): exact-match count**, from 0 to 15. One point for each of the 15 printed
  numbers that the window reproduces at the printed precision (the computed value rounds to
  the printed value): for each index the mean (6 decimals), SD (9 decimals), minimum and
  maximum (9 decimals), skewness (9 decimals), kurtosis (8 decimals for Nifty, 7 for Bank),
  and N (exact); plus the correlation (3 decimals). A second count E1 accepts anything within
  one unit of the last printed digit (this also covers truncation instead of rounding); E1 is
  reported, not used for ranking.
* **D (tie-break): distance**, the root mean square of 11 relative errors: for each index
  the log ratio of SD, the relative error of the minimum and of the maximum, the relative
  error of the skewness, the log ratio of the kurtosis; and the absolute error of the
  correlation. D is 0 for a perfect fit, but the rounding of the printed correlation
  (3 decimals) alone leaves D near 1e-4 even for the true window.
* **Ranking:** by E (higher is better), then by D (lower is better).

**Stage 2** (top candidates only): for the 20 best windows of each of the 10 best definitions,
and for the stated windows (Section 10) under the winning definition, the returns are
rebuilt directly from prices (a separate code path that also checks the fast window sums),
and the Anderson-Darling normality statistic is added for each index (printed to 3 decimals;
variants: SD with N - 1 or N, with or without the small-sample factor 1 + 0.75/N + 2.25/N^2).
This gives E2, from 0 to 17. The final answer is the stage 2 window with the highest E2,
ties broken by the stage 1 D. The ADF statistic (Table 1) is reported for the top three
windows and the stated windows but not scored, because the paper does not say which lags
or deterministic terms it used. Also reported, not scored: the number of Wednesdays and
Thursdays in the window (Table 5 suggests 13 of each if a day has 360 bars) and the median
per-minute Rogers and Satchell volatility (Table 6 prints 0.00024 and 0.00034).

**Anchor search:** separately, every minute in the whole history whose return equals a printed
before minimum or maximum (to within one unit of the 9th decimal) is listed, under every
definition. If the printed extremes exist in our data, this shows on which dates they fall,
independent of the window search.

## 9. What counts as an identification

Competitors of the winner are all windows with a different start or end date (under any
definition). The same dates under a different definition are not a competitor.

* **Exact identification:** the winner reproduces all four printed extremes exactly, and
  at least 4 of the 6 higher-moment numbers (SD, skewness, kurtosis for both indices)
  at printed precision, and every competitor has E at least 2 lower.
* **Strong identification:** the winner's E is at least 2 above every competitor's, or
  1 above with the best competitor's D at least 5 times the winner's.
* **Weak identification:** anything less. Then the answer is reported as a set: all
  windows with E at least E_best - 1 and D at most 2 x D_best.
* **Not identified:** if the best window reproduces none of the four extremes and no SD,
  skewness or kurtosis value.

## 10. Uncertainty and robustness to report

* The margin over the runner-up: E and D of the best window that differs only slightly
  (the next-ranked window), and of the best "distinct" window (one sharing less than half
  of the winner's days).
* Robustness to the return definition: the best window under each of the 204 definitions,
  and whether the winning window stays on top when switching log and simple returns,
  the overnight treatment, the session and the calendar.
* The null reference: the distribution of E over all windows of the winning definition.
* **How the stated window scores**, under the winning definition and under every definition:
  (a) the first trading day of November 2023 to the last trading day of February 2024,
  (b) the first 64 trading days from the first trading day of November 2023,
  (c) the last 64 trading days to the last trading day of February 2024,
  each with E, D and its rank among all windows.

## 11. Secondary check (corroboration only)

The same machinery is run on the after window's printed numbers (Table 1, printed with fewer
digits), under the winning definition, and the best after windows are reported. This is
not used to choose the before answer.

## 12. What would change the plan

Nothing is changed after seeing real data without being labelled post hoc. If the
contiguous search fails (for example, no window reproduces any extreme), any further idea
(for example a window with one day missing, or another price field) is reported as a
clearly labelled post hoc analysis, separate from the blind answer.

## 13. Files

* `experiments/p108_blind_window/printed_values.csv`: transcribed numbers.
* `experiments/p108_blind_window/search.py`: the search (numpy, pandas, scipy; statsmodels
  only for the reported ADF). Run with
  `OMP_NUM_THREADS=1 python experiments/p108_blind_window/search.py --raw data/raw --out results/p108_blind_window --plan-commit <hash>`.
* `tests/test_p108_blind_window.py`: SYNTHETIC tests only. They check the moment formulas
  against scipy and pandas, the rounding rule, that the fast window sums equal a direct
  computation (and an independent pandas construction), and that a window planted in
  synthetic data under one definition is recovered (right dates, right definition, E = 15,
  E2 = 17) against nearby definitions.
* Outputs (phase 2) in `results/p108_blind_window/`: `best_by_definition.csv`,
  `top20_by_definition.csv.gz`, `all_windows_winning_definition.csv.gz`,
  `top_candidates.csv` (stage 2), `stated_windows.csv`, `extreme_anchors.csv`,
  `E_distribution_top_definitions.csv`, `after_window_check.csv`, `run_info.json`
  (plan commit, input hashes, versions, run time).
