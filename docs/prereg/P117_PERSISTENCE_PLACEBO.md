# P117 pre-registration: is the fall in intraday persistence larger than in comparable windows without the expiry change?

**Status.** Written on 3 November 2026 (phase 1) by an AI agent (Claude) working for Shreyan Thanki. **This design
is committed to git before any outcome of this study has been computed.** No placebo window, no after window on the
placebo grid, no Fin Nifty model and no test statistic of this study has been computed on real data. The only
real-data computations so far are the phase-1 checks of Section 13: counts of trading days and of placebo pairs
(dates only), Fin Nifty's data coverage and expiry calendar (dates only), and the reproduction of numbers that P114
already committed (P114's rolling-window fits and P114's bootstrap draws, refitted through this study's code). The
script and its tests were also run end to end on a simulated market (SYNTHETIC). The commit that adds this file is
the time stamp; phase 2 (the run) starts only after that commit exists and has been pushed. The code that phase 2
runs (`experiments/p117_persistence_placebo.py`) and its tests (`tests/test_p117_persistence_placebo.py`, synthetic
data only) are in the same commit. Phase 2 writes `results/p117_persistence_placebo/` and
`docs/P117_PERSISTENCE_PLACEBO.md`. All results will be REAL DATA.

## In plain words

The anchor paper (Chhimwal, Pandey and Pandey 2025, "CPP") reports that after Bank Nifty's expiry day moved
(March 2024), a burst of intraday volatility dies out much faster. Our replication finds the same with the paper's
model, and P114 found that the fall also survives a better model (mcsGARCH with P109's fixes), for Nifty, on the
before window that reproduces the paper's numbers: the half-life of a shock at 12:15 falls from 141 to 18 minutes
(`docs/P114_PERSISTENCE_FIXED_MODEL.md`).

But P114 also found that this before window (14 November 2022 to 10 February 2023) is the most persistent of 44
rolling Nifty windows since November 2022. So the fall could simply be a return from an unusually persistent period
to an ordinary one, something that would have happened without any change to the expiry calendar.

**This study asks: is the before-to-after fall larger than the changes seen between comparable windows that do not
straddle the expiry change, and larger than the change in an index whose calendar did not change?**

* **Placebo in time.** We take every pair of windows with the same lengths (64 and 59 trading days) and the same
  distance between them (258 trading days) as the actual pair, slide the pair one day at a time over all the data
  from 2018 to 2026, keep the pairs that do not contain the expiry changes, and measure the change in each. If falls
  as large as the actual one are common in those pairs, the actual fall is no evidence that the expiry change caused
  it.
* **Difference-in-differences.** We measure the same change for Fin Nifty (Nifty Financial Services), whose own
  expiry calendar did not change between November 2022 and November 2024, over the same days. If Fin Nifty's
  persistence fell just as much, the fall is market-wide, not tied to the Bank Nifty change.

## Prior knowledge (what was known when this was written)

All REAL DATA, from committed files.

* **The actual changes** (`results/p114_persistence_fixed_model/persistence_change.csv`; replication's two-index
  grid). Half-life of a 12:15 shock, before (inferred window) to after, in minutes: Nifty reference 358.6 to 72.4
  (log fall 1.600), fixed model F 141.1 to 17.9 (log fall 2.064); Bank Nifty reference 162.0 to 38.0 (1.451), F
  62.7 to 10.5 (1.789). For the reference model the half-life is ln(0.5) / ln(alpha + beta).
* **P114's rolling windows** (`rolling.csv`, exploratory there): 64-day windows every 21 trading days from 14
  November 2022. Under F the Nifty half-life has median 25.1 minutes over 44 windows; the inferred before window
  (window 0, 141.0) is the largest; the three windows ending inside the after window have 24.9, 19.5 and 21.0. The
  reference's alpha + beta is at its upper bound 0.99999 (half-life about 69,000 minutes, "no decay found") in 6
  Nifty windows (around the June 2024 election and in early 2025) and 3 Bank Nifty windows. For Bank Nifty under F,
  window 0 (62.2) is the fourth largest of 38. **Nothing is known about any window before November 2022**, and no
  before-after pair has been computed other than P114's.
* **The calendar changes** (`results/event_study/events.csv`, `results/referee_expiry/calendar_changes.csv`,
  `docs/EVENT_STUDY.md` Table 1): Bank Nifty weekly expiry Thursday to Wednesday (event a, last old expiry 24 August
  2023), Bank Nifty monthly Thursday to Wednesday (event b, CPP's event, last old expiry 29 February 2024), Bank
  Nifty weeklies stop with SEBI's package (event c, last weekly 13 November 2024), Bank Nifty monthly Wednesday to
  last Thursday (c2, last old expiry 24 December 2024), Nifty weekly Thursday to Tuesday and both monthlies to
  Tuesday (d, last old weekly 21 August 2025), closing auction (e, 3 August 2026). Nifty weekly options are reported
  to have started in February 2019; the event study starts its long sample on 4 March 2019 for that reason (not
  verifiable from our files).
* **The actual pair straddles two of these changes, a and b.** The inferred before window lies before event a, so
  the comparison covers both Bank Nifty moves (as `docs/REPLICATION_NSE.md`, question 1, says).
* **Fin Nifty.** Its minute bars (`finnifty_1min.csv`, P104) cover 4 August 2021 to 27 January 2026 with a data
  hole from 28 January to 1 August 2022, and cover every day of both actual windows (Section 13). Its F&O
  bhavcopies (2 January 2023 onward) show weekly and monthly expiries on Tuesday (Monday when Tuesday is a holiday)
  up to 19 November 2024, then monthly only (last Tuesday to December 2024, last Thursday from January to August
  2025, Tuesday from September 2025). For November and December 2022 there are no Fin Nifty bhavcopies in the
  repository; its Tuesday expiry since its weekly options started in January 2021 is reported
  (`docs/REFEREE_RESPONSE_EXPIRY.md`), not verified from our files for those two months.

## 1. The statistic

For a pair of windows (a before window and an after window) and a model, the **change** is

    Delta = ln(min(HL_after, 100,000)) - ln(min(HL_before, 100,000)),

where HL is P114's primary persistence measure: the half-life, in trading minutes, of the impulse response to one
standardised shock at 12:15 (minute 180 of the 360-minute session), computed exactly from the model's recursion
(`niftygex.volatility.ext_persistence_summary`, unchanged). For the reference model it equals ln(0.5) /
ln(alpha + beta), the paper's half-life. Infinite half-lives are capped at 100,000 minutes, as in P114. **A negative
Delta is a fall** (shocks die out faster in the later window); exp(-Delta) is the fall factor. Delta is computed
for two models, each exactly as in P114 (`p114_persistence_fixed_model.fit_window`, unchanged):

* `ref`: the replication's ARMA(1,1)-mcsGARCH(1,1)-t, daily level from an AR(1)-EGARCH(1,1)-t (Table 2's model);
* `F`: P114's fixed model, X1 (HAR daily level) + X2 (overnight decay) + X3 (today's running mean).

Every window is fitted in sample, with its daily levels fitted on all days from the start of the data up to the
window's last day (as the replication and P114 do).

## 2. The actual pair

The inferred before window (14 November 2022 to 10 February 2023, 64 days) and the replication's after window
(1 March to 31 May 2024, 59 days), Nifty primary, Bank Nifty secondary. On each index's own grid there are **258
trading days strictly between the two windows** (the gap), so a pair spans 64 + 258 + 59 = 381 trading days.

**For the placebo test the actual pair is fitted by exactly the same code, on the same grid, as every placebo pair:**
each index's one-index grid (days with at least 300 one-minute bars from 09:15 to 15:14; P114's grid for its
rolling windows), bars loaded to 2 November 2026 (Nifty) and 22 April 2026 (Bank Nifty). On this grid the before
window is P114's rolling window 0 (committed: Nifty reference 359.0, F 141.0; Bank Nifty 164.9 and 62.2 minutes),
and the after window is fitted in phase 2. The actual Delta of the placebo test is this one-grid value. P114's
Table-2-grid values (Section "Prior knowledge") are reported next to it; they differ only through the days used for
the daily levels.

## 3. The placebo pairs

* **Every pair** on an index's grid: before window = grid rows s to s + 63, after window = rows s + 322 to s + 380
  (64 + 258 = 322), for every start s (one-day steps) with at least **250 grid days of data before s** (so that the
  daily EGARCH and HAR are fitted on at least a year of daily data) and the after window inside the data.
* **Straddling.** A pair straddles a calendar change if its before window starts before the change's date and its
  after window ends on or after it. The change dates (first day on which the old schedule no longer applied: the
  day after the last expiry on the old schedule; for the Nifty weeklies the event study's long-sample start; for
  the closing auction its first day):

| Change | Date used | What changed |
|---|---|---|
| Nifty weeklies | 4 March 2019 | Nifty weekly options (reported to start in February 2019) |
| a | 25 August 2023 | Bank Nifty weekly expiry Thursday to Wednesday |
| **b (CPP's event)** | **1 March 2024** | Bank Nifty monthly expiry Thursday to Wednesday |
| c | 14 November 2024 | Bank Nifty weekly options stop (SEBI package of 20 November 2024) |
| c2 | 25 December 2024 | Bank Nifty monthly expiry Wednesday to last Thursday |
| d | 22 August 2025 | Nifty weekly expiry Thursday to Tuesday; both monthlies to Tuesday |
| e | 3 August 2026 | closing auction sets the official close |

* **Three placebo sets**, computed for both indices and both models:

| Set | Pairs kept | Role |
|---|---|---|
| **primary** | pairs that straddle **neither a nor b** | primary (Nifty), S1 (Bank Nifty) |
| clean | pairs that straddle **no change in the table** | secondary S2 |
| literal | pairs that do not straddle **b** (the task's literal definition) | secondary S3 |

**Why the primary set excludes a as well as b.** The actual pair straddles both Bank Nifty moves, so a pair that
straddles a is partly treated by the same treatment; and the a-only pairs are exactly the pairs whose before window
lies next to, or on, the actual before window, which would bias the comparison toward "normal". The literal set
(only b excluded) is reported as S3 so that this choice can be seen.

**Why the primary set does not exclude every calendar change (the clean set is secondary).** We registered both and
chose in advance, from the dates alone (Section 13):

* With pairs 381 trading days long, the clean set can only use 2019 to 2023, and it holds **only 1 to 2 pairs with
  wholly separate spans and about 5.3 pairs that share no trading day** (724 Nifty pairs, all overlapping). That is
  too little to describe what changes "normally" look like, and it falls below the minimum of Section 4.
* The primary set holds 1,223 Nifty pairs (before windows from April 2018 to April 2025) with **3 to 4 wholly
  separate spans and about 12.4 day-disjoint pairs**, and it covers the 2024 to 2026 market as well as 2018 to 2023.
* **Many primary pairs straddle other changes** (Section 13). Nifty: of 1,223 primary pairs, 224 straddle the
  Nifty weekly launch, and all 275 pairs that start after event b straddle event d (172 also c, 199 c2, 63 e); only
  724 straddle no change at all (they are the clean set). Bank Nifty: of 1,152, 286 straddle the Nifty weekly launch
  and all 143 pairs after event b straddle c, c2 and d; 723 straddle none. If those changes also
  shorten persistence, falls become more common among the placebo pairs, so the primary test is **conservative for
  the verdict "the fall is unusual"** and may be lenient for "within the normal range". The clean set (S2) shows the
  same comparison without them, even though it gives no verdict.

**Why the gap is not shortened.** Persistence drifts slowly, so the change between two windows depends on how far
apart they are. A shorter gap would make placebo changes smaller and the test anti-conservative. The fallback
below is used only if the primary set misses the minimum, which by the dates it does not.

## 4. The placebo p-value and how overlap is handled

* **Primary p** (each set): p = (1 + number of placebo pairs with Delta at or below the actual Delta) / (1 +
  number of placebo pairs), one-sided (the actual fall is unusually large). Pairs whose Delta cannot be computed are
  dropped and counted.
* **Overlap.** Neighbouring pairs share most of their days, so they are not independent. Overlap does not bias the
  share: under a stationary process, the empirical distribution of all overlapping pairs estimates the distribution
  of the change between two such windows (the idea of subsampling: Politis, Romano and Wolf 1999, *Subsampling*,
  Springer; and of Andrews 2003, "End-of-sample instability tests", *Econometrica* 71(6), 1661 to 1694, which
  compares one statistic with the same statistic on all overlapping subsamples). Using every pair is more efficient
  than keeping only non-overlapping ones. What overlap does is **lower the resolution**: the information is closer
  to the number of separate periods than to the number of pairs. We account for this in three registered ways:
  1. **A conservative count (S4): only pairs that share no trading day.** For each phase 0 to 380, walk the pairs
     of the set in order of their start and keep a pair if it shares no trading day with any pair kept so far
     (pairs whose windows interleave, for example one pair's before window inside another pair's gap, share no day
     and can both be kept). p_phase = (1 + kept pairs at or below the actual) / (1 + kept pairs); the S4 p is the
     mean over the 381 phases. With about 12 day-disjoint pairs it **cannot fall below about 0.08**, so it can
     confirm "within the normal range" but can never show "unusual" at 5%; it is reported for that reason.
  2. **A minimum for a verdict.** A set gives a verdict only if it has **at least 200 placebo pairs and on average at
     least 8 day-disjoint pairs** (over phases), so that the null distribution is drawn from at least about 8 x 123
     = 1,000 distinct trading days. Otherwise its p-values are reported with "no verdict: too few separate placebo
     periods". By the dates (Section 13) the primary and literal sets meet the minimum for both indices and the
     clean set does not.
  3. **The number of wholly separate spans** (pairs whose 381-day spans do not overlap at all) is reported with
     every set, so that the reader sees how few truly independent periods exist.
* **Fallback** (fixed now, not expected to be needed): if the primary set of an index misses the minimum (for
  example because fits fail), the literal set takes its role for that index; if that also misses it, the literal
  set with gap 0 (adjacent windows), reported as a deviation.

## 5. The conditional placebo (S5)

P114's worry is "regression to the mean": if the before window was unusually persistent, any later window is likely
to be less persistent. S5 asks whether the actual after window is unusually short **given** how persistent the
before window was. Over the primary set, fit by OLS Delta = c0 + c1 ln(HL_before) + e; the actual pair's residual
e_act = Delta_act - (c0 + c1 ln(HL_before,act)); p = (1 + placebo residuals at or below e_act) / (1 + pairs). A low
p means the after window is shorter than the placebo pairs predict from the before window's persistence. If the
primary test says "unusual" but S5 says "within the normal range", the fall is what usually follows such a
persistent before window. (If the actual before half-life lies beyond the placebo range, S5 extrapolates; the ranks
of the actual before and after half-lives among the placebo windows are reported to show this.)

## 6. The difference-in-differences against Fin Nifty (secondary family D)

* **Control index: Fin Nifty.** It is the only index besides Nifty and Bank Nifty with one-minute bars for both
  windows. Its own expiry calendar did not change between November 2022 and November 2024 (Prior knowledge). Its
  expiry weekday (Tuesday) also did not change its overlap with Bank Nifty's: it coincided with neither Bank
  Nifty's old Thursday nor its new Wednesday.
* **Statistic:** DiD = Delta_treated - Delta_FinNifty over the actual windows (the same 64 + 59 dates; P114's
  two-index grid dates for the treated index), for each model; treated = Nifty and Bank Nifty. Negative DiD: the
  treated index's half-life fell more than Fin Nifty's.
* **Fin Nifty's fit.** Exactly P114's `fit_window` on Fin Nifty's own one-index grid, restricted to the same dates
  (checked: identical dates in both windows). Its daily EGARCH level is fitted on all Fin Nifty days from 4 August
  2021 up to the window's last day, **with the single daily return that spans the data hole (28 January to 1 August
  2022) left out and the two sides joined** (`egarch_level_joined`); without a hole this is exactly the
  replication's function (tested). The HAR level of F uses the grid days in order (its lags join over the hole,
  which affects only August 2022, months before the before window). 177 Fin Nifty grid days precede the before
  window (Nifty: over 1,300), so Fin Nifty's daily levels are less well estimated.
* **Inference: the day-block bootstrap of P114, paired.** For the treated index we use P114's committed draws
  (`results/p114_persistence_fixed_model/bootstrap_draws.csv`, 400 per window, stored to 12 significant digits).
  Fin Nifty is resampled with **the treated index's seeds** (P114's seed index for that window and index, 1000 i +
  chunk), so each Fin Nifty draw uses **the same resampled days** as the treated draw of the same number (same window
  length, same random stream; tested on simulated data, checked on the real draws in Section 13). On each resample
  Fin Nifty's diurnal curve is re-estimated and both models refitted, as in P114. Fin Nifty is therefore bootstrapped
  twice per window (with Nifty's and with Bank Nifty's seeds). Before and after draws are paired by draw number, as
  in P114. Statistic: DiD per draw; its 95% percentile interval; one-sided p = (1 + draws with DiD at or above 0) /
  (1 + draws).
* **No placebo distribution of the DiD.** The only Fin Nifty pairs that do not straddle b start after March 2024,
  and all of them straddle Fin Nifty's own changes (its weeklies stopped on 19 November 2024; its monthly moved to
  Thursday in January 2025). There is no clean placebo for the control, so the DiD is tested by the bootstrap only.

## 7. Hypotheses, tests and decision rule

**Primary family P (2 tests, Holm over 2).** Nifty, primary set, Delta under `ref` (`P_nifty_primary_ref`) and
under `F` (`P_nifty_primary_F`); placebo p of Section 4.

**Decision rule** (applied by `verdict` in the script to each model's Holm-adjusted p, separately):

| Outcome | Rule | What it means for CPP's claim that the expiry change made shocks die out faster |
|---|---|---|
| **the fall is unusual** | Holm p at most 0.05 and the set meets the minimum | the fall is larger than about 95% of changes between comparable windows that straddle neither Bank Nifty move; consistent with the causal reading (not proof: one event, and many other things changed between February 2023 and March 2024) |
| **unclear** | Holm p above 0.05 and at most 0.10 | the data cannot tell the fall apart from ordinary drift |
| **the fall is within the normal range** | Holm p above 0.10 | falls this large are common without these calendar changes; the before-after fall is no evidence that the change caused faster decay, and is consistent with a return from an unusually persistent period |
| no verdict | the set misses the minimum | too few separate placebo periods |

**How the two models are read together** (registered): both "within the normal range": the paper's persistence
result gives no causal evidence. `ref` "unusual" but `F` not: the unusual size depends on the misspecified model.
`F` "unusual": the better model sees an unusually large fall; then S5 says whether the before window's persistence
explains it, and D whether Fin Nifty fell as much.

**Secondary families** (Holm within each; they cannot change the primary verdicts; the same rule labels each row):

* **S1 (2 tests)**: Bank Nifty, primary set, `ref` and `F`.
* **S2 (4 tests)**: clean set, Nifty and Bank Nifty, `ref` and `F` (by the dates: "no verdict", p reported).
* **S3 (4 tests)**: literal set (only b excluded), Nifty and Bank Nifty, `ref` and `F`.
* **S4 (4 tests)**: the day-disjoint p of Section 4 on the primary set, Nifty and Bank Nifty, `ref` and `F`.
* **S5 (4 tests)**: the conditional placebo of Section 5 on the primary set, Nifty and Bank Nifty, `ref` and `F`.
* **D (4 tests)**: DiD against Fin Nifty, Nifty and Bank Nifty, `ref` and `F`. Rule on the Holm p: at most 0.05
  "the fall is larger than the control's"; above 0.10 "the fall is not clearly larger than the control's"; between:
  "unclear". If the treated index's fall is not larger than Fin Nifty's, the fall is market-wide (or shared by a
  closely related index) and not specific to the index whose expiry calendar is tied to the change.

**Exploratory (no adjustment)**: **E1 (2 tests)** Fin Nifty's own change (bootstrap with Nifty's seeds; p for "its
half-life fell"), `ref` and `F`. **Descriptive, no p**: the quantiles of the placebo changes; the share of placebo
pairs with a fall; the percentile ranks of the actual before and after half-lives among the placebo before and after
windows; the placebo changes plotted against time; the Fin Nifty draws with Bank Nifty's seeds summarised.

Every p-value goes in `results/p117_persistence_placebo/tests.csv` with the columns `test_id, hypothesis, role,
sample, estimate, statistic, p_raw, sidedness, p_method, p_adjusted, adjust_method`. Any test added after this commit
is labelled "post hoc".

**What we expect** (stated in advance; it changes no rule): P114's rolling windows show that under F the actual
before window is the most persistent since November 2022, and that the reference model's half-lives swing over
several orders of magnitude (it hits its bound in 9 windows). We guess that under `ref` the fall is "within the
normal range" (huge placebo swings), that under `F` it is "unclear" or "unusual" on the unconditional test but
"within the normal range" on the conditional one (S5), and that Fin Nifty's half-life also fell (D "not clearly
larger"). This is a guess, not a rule.

## 8. The exact commands

From the repository root, at the commit that adds this file, with the raw data listed in `data/MANIFEST.csv` in
`data/raw`. Work files go to `data/processed/p117/` (git-ignored); `windows` resumes where it stopped.

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p117_persistence_placebo.py inputs
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p117_persistence_placebo.py windows --index nifty
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p117_persistence_placebo.py windows --index banknifty
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p117_persistence_placebo.py control
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p117_persistence_placebo.py analyze --plan-commit <this commit>
```

One worker process. `windows` fits the actual pair first, then every window that a placebo pair needs (1,351
before and 1,351 after windows for Nifty, 1,280 and 1,280 for Bank Nifty), both models, about 0.6 to 0.9 seconds
per window (about 70 minutes for both indices). `control` refits the treated windows on the two-index grid as a
check, fits Fin Nifty and runs its four bootstrap keys (400 draws each). `analyze` writes `tests.csv`,
`verdict.csv`, `placebo_summary.csv` (every index, model and set: counts, quantiles, p-values, day-disjoint and
separate-span counts, the conditional fit, ranks), `placebo_pairs.csv` (every pair with its half-lives, changes and
set flags), `windows_nifty.csv` and `windows_banknifty.csv` (every window fit), `did.csv`, `control_fits.csv`,
`control_bootstrap_draws.csv`, `reproduction_p114_rolling.csv`, `reproduction_p114_control.csv`, `fig_placebo.png`
(the placebo distributions with the actual change marked), `fig_placebo_time.png` and `run_info.json`.

## 9. Checks inside the run

* `inputs` refuses to continue unless the five raw files (Nifty, Bank Nifty and Fin Nifty minute bars and their
  source files) match `data/MANIFEST.csv`.
* `windows` refuses to continue unless the actual windows have 64 and 59 days on the grid; it compares every one of
  P114's committed rolling windows that coincides with a before window here (`reproduction_p114_rolling.csv`).
* `control` refuses to continue unless Fin Nifty has exactly the treated windows' dates; it compares the treated
  point fits with P114's `fits.csv` and the first three draws of chunk 0 of each treated key with P114's committed
  draws (`reproduction_p114_control.csv`). A mismatch is reported as a deviation.

## 10. Deviations policy

* After this file is committed the design does not change. Any change is a deviation, listed with its reason and
  with whether any outcome had been seen when it was decided. No specification search: no other gap, window length,
  history rule, set, shock minute, model or measure is promoted after outcomes are seen.
* A window whose fit raises an error is recorded as not computed (registered in the code); its pairs are dropped and
  counted. A fit that does not converge is used as it is (as P114).
* A crash is fixed with the smallest change that lets the run finish, and only if it changes no number already
  computed.
* If the windows are not finished by 18:00 UTC on 3 November 2026, `analyze` is run on the windows done (the
  primary Nifty pairs come first in time order with Bank Nifty after), and the rows that lack them are reported as
  not computed.

## 11. Known weaknesses

1. **Few independent periods.** A pair spans 381 trading days (about a year and a half), so the 2018 to 2026 data
   hold only 3 to 4 wholly separate spans; the placebo p is a share over overlapping pairs and should be read as
   "where the actual change sits among the changes that occurred", not as a test with many independent draws.
2. **The placebo pairs straddle other changes** (Section 3), and they include the COVID-19 crash of 2020 and the
   June 2024 election. These are part of what "ordinary" changes look like over a year; they are not removed.
3. **One event.** Even an "unusual" fall does not prove that the expiry change caused it: the actual pair spans a
   year in which much else changed (for example index option volumes, lot sizes and Sensex's relaunched weeklies).
4. **The control is related to the treated indices.** Fin Nifty shares its largest constituents with Bank Nifty
   (and, less so, with Nifty). If the Bank Nifty change affected bank stocks in general, Fin Nifty is partly
   treated and the DiD is biased toward zero. Its daily levels rest on 177 days of history across a data hole, and
   its November and December 2022 bars are not validated against official values (P104's validation starts in 2023).
5. **The reference model's half-life is unstable**: at its bound in several windows (a half-life of about 69,000
   minutes, capped at 100,000 for logs), which makes its placebo changes heavy-tailed. The p-values are ranks, so
   these values count as very large, not as exact numbers.
6. **In sample, one grid per index.** Each window's parameters come from the window itself; the one-index grid
   differs from the replication's two-index grid only before the actual windows (P114, Section 12 of its plan).
7. **The fixed model's night cap** (P114): when delta is small, a 12:15 shock cannot have a half-life much beyond
   the close (179 minutes later); this compresses F's long half-lives in placebo and actual windows alike.

## 12. Code

* `experiments/p117_persistence_placebo.py`, new: stages `inputs`, `windows`, `control`, `analyze`; `--synthetic`
  runs all of them on a simulated three-index market (labelled SYNTHETIC; the third index has a data hole). It
  imports P114's script (unchanged) for the grids, the window fit, the daily levels and the bootstrap step, and the
  replication's script through it. **No shared code in `src/niftygex/` is changed or added**, so every earlier result
  is untouched.
* `experiments/p117_plan_checks.py`, new: the checks of Section 13.
* `tests/test_p117_persistence_placebo.py`, new, synthetic data only (16 tests): the gap; the straddle rule and the
  three sets; the fast shared-day test against a brute-force check; the greedy day-disjoint selection; the placebo,
  day-disjoint, separate-span and conditional calculations on hand-made numbers; the decision rules; the half-life
  cap; the daily EGARCH joined over a hole (and unchanged without one); the bootstrap with an explicit seed against
  P114's bootstrap, draw for draw; an end-to-end dry run of every stage.

## 13. Checks run before this file was committed (3 November 2026)

Output of `PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p117_plan_checks.py`, committed as
`results/p117_persistence_placebo/plan_checks_stdout.txt`. REAL DATA dates and committed numbers only.

* **Grids**: Nifty 2,360 days (3 April 2017 to 2 November 2026), Bank Nifty 2,289 (2 January 2017 to 22 April 2026),
  Fin Nifty 966 (4 August 2021 to 27 January 2026). On both treated grids the gap between the actual windows is 258
  days, and the actual pair straddles a and b only.
* **Placebo pairs** (dates only): the file gives the counts of each set, the day-disjoint and separate-span counts,
  how many primary pairs straddle each other change, and the windows to fit (1,351 + 1,351 Nifty, 1,280 + 1,280 Bank
  Nifty). In short: primary 1,223 Nifty and
  1,152 Bank Nifty pairs (mean day-disjoint 12.40 and 11.20; 3 to 4 separate spans; minimum met); clean 724 and 723
  (5.29 and 5.28; 1 to 2; minimum not met); literal 1,350 and 1,279 (13.92 and 12.42; 3 to 4; met).
* **P114's rolling windows reproduce**: windows 0, 22 and 43 (Nifty) and 0 and 37 (Bank Nifty), both models: alpha,
  beta, gam_x, delta, nu and the half-life identical to the committed `rolling.csv`.
* **P114's bootstrap draws reproduce** through this study's bootstrap with explicit seeds: the first two draws of
  chunk 0 of each of the four treated keys, both models (alpha, beta, gam_x, delta and the half-life), 20 of 20
  values per key identical to the 12 significant digits of the committed `bootstrap_draws.csv` (the file is rounded
  to 12 digits; in memory the values printed agree).
* **Fin Nifty**: both actual windows have exactly the same 64 and 59 dates as Nifty; one data hole (28 January to 1
  August 2022); 177 grid days before the before window; expiry calendar as in "Prior knowledge".
* **Dry run** (SYNTHETIC, three simulated indices, 240 days, the third with a 12-day hole): all stages ran.
* **Tests**: `PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_p117_persistence_placebo.py`: 16 passed.
