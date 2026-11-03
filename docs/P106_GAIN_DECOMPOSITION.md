# P106: where does the forecasting hybrid's gain come from?

Study P106, run by an AI agent (Claude) working for Shreyan Thanki on 3 November 2026.
**All numbers below are REAL DATA** (Nifty, Bank Nifty and Fin Nifty, NSE data as in the headline runs) unless
marked SYNTHETIC. Every number comes from a file in `results/p106_gain_decomposition/` (file named next to each
table) or, where marked, from an earlier committed results file.

* Registered plan: `docs/prereg/P106_GAIN_DECOMPOSITION.md`, commit `71bb3b6b922540021082a40556cf8d915a45f085`
  (committed 06:01:23 UTC, before any loss of an ablated model was computed).
* Code: `experiments/p106_gain_decomposition.py`, one new code path in `src/niftygex/forecast.py`
  (`WalkForwardConfig.drop_sets`, empty by default). Tests (SYNTHETIC only): `tests/test_p106_gain_decomposition.py`.
* Every p-value of the study: `results/p106_gain_decomposition/tests.csv` (189 rows).

*Note added 3 November 2026 after study P116:* here "mcsGARCH" and "the anchor paper's model" mean our
implementation (a separate time-of-day curve for expiry days, a 5-minute smoothing of the curves, a zero mean), not
the paper's own specification. Against the paper's specification (one curve, no smoothing, ARMA(1,1) mean) see
`docs/P116_BENCHMARK_AS_PRINTED.md`; there the hybrid's gain at 15 and 30 minutes is 4.43% and 6.64% on Nifty, 4.73%
and 6.20% on Bank Nifty and 4.03% and 5.18% on Fin Nifty. The contributions and shares below are measured in points
of our implementation's QLIKE and against its gain, and were not recomputed in P116.

## 1. The answer in plain words

Our hybrid forecasts how much the Nifty will move in the next 15 or 30 minutes. It starts from the anchor
paper's model (mcsGARCH) and multiplies its forecast by a correction learnt by LightGBM from 41 inputs. Its
loss (QLIKE) is 5.80% lower than mcsGARCH's at 15 minutes and 7.65% lower at 30 minutes. We sorted the 41
inputs into five groups and refitted the hybrid without each group in turn, changing nothing else.

**Question 1: which inputs carry the gain?**

* **Two groups carry it: recent realized volatility and the time of day.**
  * Removing the **realized-volatility history** (12 inputs: realized variance over the last 15, 30 and 60
    minutes, yesterday, last week, and so on, plus mcsGARCH's own forecast) costs **2.00 points at 15 minutes
    and 2.90 points at 30 minutes**, that is **34% and 38% of the hybrid's gain**.
  * Removing the **time of day** (4 inputs: where the window sits in the session) costs **1.34 and 1.92
    points, 23% and 25% of the gain**.
  * Both are "supported" under the registered rule at both horizons: Holm-adjusted p below 0.01, and all
    five sets of random seeds agree.
* **The option market, India VIX and the calendar add nothing on their own.** Removing the 16 option-implied
  inputs changes the loss by +0.11 and -0.29 points; removing VIX by -0.19 and -0.27; none is significant, and
  these changes are within what changing LightGBM's random seeds alone can produce (flagged "within seed
  noise"). Removing the 8 calendar inputs makes the hybrid slightly *better* (-0.41 and -0.58 points; the 95%
  interval of its share of the gain, -14% to -1% and -15% to -1%, lies below zero).
* **A hybrid that sees only what mcsGARCH itself sees** (past one-minute returns, the time of day and the
  expiry-day flag) **gets all of the gain**: it is 6.06% and 8.01% better than mcsGARCH, against 5.80% and
  7.65% for the full hybrid. So the gain comes from the method (a learnt correction of mcsGARCH using
  mcsGARCH's own kind of information), not from extra information about the option market.
* The five groups' shares add up to only about half of the gain (49% at 15 minutes and 48% at 30 minutes).
  The rest is shared: when one group is removed, others can partly stand in for it.

**Question 2: why does the hybrid beat the fairer benchmark only since SEBI's November 2024 cut?**

The fairer benchmark is `mcsgarch_iv`, an mcsGARCH whose daily level comes from the option market.

* **The edge over `mcsgarch_iv` really did grow** after 20 November 2024: from 0.21 to 7.18 points at
  15 minutes and from 0.64 to 8.86 points at 30 minutes (points of mcsGARCH's QLIKE; p 0.032 and 0.004,
  Holm-adjusted 0.032 and 0.009). In % of `mcsgarch_iv`'s loss: from -0.22% to -7.10%, and from -0.68% to
  -8.80%.
* **Mostly because the benchmark got worse, not because the hybrid got better.**
  * `mcsgarch_iv` went from 4.98 points *better* than mcsGARCH to 1.06 points worse at 15 minutes (a change
    of 6.04 points), and from 6.06 better to 0.64 worse at 30 minutes (6.70 points).
  * The hybrid's own gain over mcsGARCH changed much less, and not significantly: 5.19 to 6.12 points and
    6.70 to 8.22 points (changes of 0.93 and 1.52, p 0.448 and 0.293).
  * The benchmark's share of the change in the edge is 0.87 at 15 minutes (95% interval 0.48 to 1.53) and 0.82
    at 30 minutes (0.20 to 1.18).
  * The benchmark's change on its own does not pass its Holm adjustment (raw p 0.034 and 0.038, Holm 0.135),
    so "mostly the benchmark" rests on the sizes and the share, not on a significant test of that piece alone.
* **The group-by-group interaction test (did any group's contribution change between the periods?):
  nothing passes the registered rule.** All 10 primary rows are "not supported".
  * The closest is the **option-implied group at 30 minutes**. It was worth +1.14 points before the cut and
    -1.16 points after it (a change of -2.30 points, p 0.003). The Holm-adjusted p is 0.051, just above
    0.05, although all five seed sets agree on the sign.
  * The drop is in the direction we registered in advance, and the robustness checks point the same way: a
    bootstrap gives Holm 0.009, and leaving out the EWMA-fallback days gives Holm 0.030.
  * But the placebo shows that this period test can call a pure seed difference significant. So we call it a
    likely but unconfirmed drop.
  * A secondary test on everything mcsGARCH lacks (option market, VIX and the calendar apart from the
    expiry-day flag, together) finds a significant drop at both horizons: -1.44 points (Holm 0.036) and
    -2.52 points (Holm 0.006), with all five seed sets agreeing.
* **Put together:**
  * The hybrid's gain comes from realized history and time of day, and it held steady across the cut.
  * The benchmark, which leans on the option market's level, lost its advantage after the cut.
  * The option-market inputs that helped the hybrid a little before the cut helped it less, or hurt it, after.
  * So the falling value of option-market information worked *against* the hybrid's edge over `mcsgarch_iv`,
    not for it. The edge appears because the benchmark lost more.

## 2. What was registered (in short)

The plan (`docs/prereg/P106_GAIN_DECOMPOSITION.md`) was committed and pushed before anything in this study was
computed on real data except a reproduction check that computes no loss (commit 71bb3b6, 3 November 2026,
06:01 UTC). In short:

* **Five groups** cover all 41 inputs of the hybrid's LightGBM correction, each input in exactly one group:

  | Group | Inputs | What it is |
  |---|---|---|
  | option-implied (`implied`) | 16 | everything computed from the previous evening's Nifty option prices: implied variance of the nearest expiry and one month out, the term structure, skew, smile, put/call ratios, implied minus realized variance, the move since the open in implied standard deviations, and the implied-level mcsGARCH's forecast |
  | India VIX (`vix`) | 1 | VIX at the previous close |
  | time of day (`intraday`) | 4 | where the window sits in the session (and its expiry-day version), and mcsGARCH's time-of-day share of the window |
  | calendar and regime (`calendar`) | 8 | expiry day or not, trading days to expiry, the expiry regime, weekday, monthly expiry, day before expiry, Bank Nifty expiry day, closing auction |
  | realized history (`realized`) | 12 | everything computed from past one-minute index returns: realized variance over the last 15, 30, 60 minutes, so far today, yesterday, last week and month, the same window yesterday, the jump share, a signed-return flow measure, and mcsGARCH's own forecast |

* **Variants.**
  * The hybrid refitted without one group (5 variants).
  * Without the option-implied group and VIX together (`optionmkt`).
  * With only what mcsGARCH itself sees (`mcsinfo`: past one-minute returns, the time of day and the expiry-day
    flag).
  * Two exploratory halves of the implied group (`implied_level`, `implied_shape`).
  * Everything else stays as in the headline run:
    * the same 942 days and 792 test days (10 August 2023 to 2 November 2026);
    * refits every 21 trading days;
    * the same LightGBM settings;
    * the same five seeds.
* **Part 1 test** (primary): for each group and for 15 and 30 minutes, does removing the group raise the loss?
  * Contribution C = 100 x (loss without the group - loss of the full hybrid) / mcsGARCH's loss, in **points of
    mcsGARCH's QLIKE**. The hybrid's whole gain is 5.80 and 7.65 points in this unit.
  * One-sided Diebold-Mariano test on daily average QLIKE: the project's version, with Newey-West variance,
    the Harvey-Leybourne-Newbold factor and Student t.
* **Part 2 test** (primary): does each group's contribution differ between two periods?
  * Period 1 is the multi-weekly period: 312 test days, to 19 November 2024.
  * Period 2 runs from 20 November 2024: 480 test days, the first on 21 November 2024.
  * The test is two-sided. Each period's contribution is a ratio of two averages; its uncertainty comes from
    the delta method with a Newey-West variance.
  * A day-block bootstrap is the robustness check.
* **One Holm adjustment over all 20 primary tests.**
* **Seeds.**
  * Every primary comparison was repeated with four more seed sets: P99's seeds 101 to 104.
  * **Supported**: the Holm-adjusted p is below 0.05 *and* all five seed sets agree on the sign.
  * **Seed-fragile**: the p passes but a seed set disagrees.
  * **Not supported**: otherwise.
  * Estimates smaller than twice their seed standard deviation are flagged "within seed noise".

The secondary (S), robustness (R), placebo (PL) and exploratory (E) families are listed in Section 4 of the
plan. Each family has its own Holm adjustment, and the exploratory family has none.

## 3. Part 1: which groups carry the gain (REAL DATA, Nifty)

Source: `tests.csv` (rows `P1_*`), `verdict.csv`, `shares.csv`, `seed_repeats.csv`. Contribution in points of
mcsGARCH's QLIKE (positive: removing the group raises the loss, so the group helps). Share = contribution /
the hybrid's gain (5.80 points at 15 minutes, 7.65 at 30), with a 95% stationary-bootstrap interval. One-sided
p. Seed SD = standard deviation of the contribution across the five seed sets.

| Group | Horizon | Contribution (points) | Share of gain [95% CI] | p | Holm p | Seed sets with removal worse | Seed SD | Verdict |
|---|---|---|---|---|---|---|---|---|
| realized | 15 min | +2.00 | 34% [27%, 43%] | below 0.001 | below 0.001 | 5 of 5 | 0.03 | **supported** |
| realized | 30 min | +2.90 | 38% [27%, 50%] | below 0.001 | below 0.001 | 5 of 5 | 0.25 | **supported** |
| time of day | 15 min | +1.34 | 23% [14%, 32%] | below 0.001 | below 0.001 | 5 of 5 | 0.12 | **supported** |
| time of day | 30 min | +1.92 | 25% [15%, 35%] | below 0.001 | 0.001 | 5 of 5 | 0.26 | **supported** |
| implied | 15 min | +0.11 | 2% [-14%, 16%] | 0.404 | 1.000 | 5 of 5 | 0.11 | not supported (within seed noise) |
| implied | 30 min | -0.29 | -4% [-14%, 7%] | 0.776 | 1.000 | 1 of 5 | 0.18 | not supported (within seed noise) |
| VIX | 15 min | -0.19 | -3% [-9%, 1%] | 0.930 | 1.000 | 2 of 5 | 0.22 | not supported (within seed noise) |
| VIX | 30 min | -0.27 | -4% [-8%, 1%] | 0.928 | 1.000 | 1 of 5 | 0.15 | not supported (within seed noise) |
| calendar | 15 min | -0.41 | -7% [-14%, -1%] | 0.948 | 1.000 | 0 of 5 | 0.12 | not supported |
| calendar | 30 min | -0.58 | -8% [-15%, -1%] | 0.970 | 1.000 | 1 of 5 | 0.24 | not supported |

So under the plan's wording, the realized history and the time of day each **carry part of the gain at both
horizons**; the implied group, VIX and the calendar carry none at either horizon.

**Across the five seed sets** (`seed_repeats.csv`, part "1 (all days)"): the realized group's contribution runs
from 1.93 to 2.02 points at 15 minutes (mean 1.98) and from 2.90 to 3.55 at 30 minutes (mean 3.25); the time of
day from 1.17 to 1.52 (mean 1.33) and from 1.92 to 2.62 (mean 2.24). The calendar group's contribution is
negative in all five seed sets at 15 minutes (mean -0.27) and in four of five at 30 minutes (mean -0.24). The
implied group's is positive in all five at 15 minutes but tiny (mean +0.12, SD 0.11), and negative in four of
five at 30 minutes (mean -0.12).

**The option market as a whole, and "method or information"** (secondary family S3, one-sided DM, Holm over 6
rows; `tests.csv` rows `S3_*`):

| Test | 15 min | 30 min |
|---|---|---|
| (a) removing the option-implied group and VIX together (`optionmkt`, 17 inputs) raises the loss | +0.60 points, p 0.160 (Holm 0.641) | -0.06 points, p 0.561 (Holm 1.000) |
| (b) the `mcsinfo` hybrid (no option market, no VIX, no calendar except the expiry flag) beats mcsGARCH | -6.06% of mcsGARCH's QLIKE, p below 0.001 (Holm below 0.001) | -8.01%, p below 0.001 (Holm below 0.001) |
| (c) removing what `mcsinfo` removes (24 inputs) raises the loss | -0.26 points, p 0.801 (Holm 1.000) | -0.36 points, p 0.802 (Holm 1.000) |

The full hybrid is -5.80% and -7.65% against mcsGARCH (`contributions.csv`), so the `mcsinfo` hybrid is a little
better than the full one. That direction was not a registered test. In all five seed sets the `mcsinfo` hybrid
had the lower loss at both horizons (`seed_repeats.csv`: contribution negative in 5 of 5, means -0.28 and -0.22
points). Removing VIX together with the chain (`optionmkt`) does not reveal a hidden option-market contribution:
+0.60 points at 15 minutes (positive in all five seed sets, mean +0.36) and -0.06 at 30 minutes, neither
significant.

## 4. Part 2: the November 2024 cut (REAL DATA, Nifty)

### 4.1 Benchmark or hybrid?

Source: `periods.csv` (subset "all days", headline seed set), `tests.csv` rows `S1_*` and `S2_*`. All in points
of the period's mcsGARCH QLIKE. G = the hybrid's gain over mcsGARCH; B = `mcsgarch_iv`'s extra loss over
mcsGARCH (negative: `mcsgarch_iv` is better than mcsGARCH); E = B + G = the hybrid's edge over `mcsgarch_iv`.
Two-sided delta-method test of the change.

| Quantity | Horizon | Period 1 (312 days) | Period 2 (480 days) | Change | p | Holm p (family) |
|---|---|---|---|---|---|---|
| E, edge over `mcsgarch_iv` | 15 min | +0.21 | +7.18 | +6.97 | 0.032 | 0.032 (S1) |
| E, edge over `mcsgarch_iv` | 30 min | +0.64 | +8.86 | +8.22 | 0.004 | 0.009 (S1) |
| B, benchmark's extra loss | 15 min | -4.98 | +1.06 | +6.04 | 0.034 | 0.135 (S2) |
| B, benchmark's extra loss | 30 min | -6.06 | +0.64 | +6.70 | 0.038 | 0.135 (S2) |
| G, hybrid's gain | 15 min | +5.19 | +6.12 | +0.93 | 0.448 | 0.586 (S2) |
| G, hybrid's gain | 30 min | +6.70 | +8.22 | +1.52 | 0.293 | 0.586 (S2) |

* The benchmark's share of the change in the edge (B's change / E's change) is 0.87 [0.48, 1.53] at 15 minutes
  and 0.82 [0.20, 1.18] at 30 minutes (95% stationary-bootstrap intervals; `periods.csv`).
* In % of `mcsgarch_iv`'s QLIKE, the unit of `results/referee_forecast/gw_group_margins.csv`, the hybrid's edge
  moves as follows:
  * at 15 minutes, from -0.22% in period 1 to -7.10% in period 2;
  * at 30 minutes, from -0.68% to -8.80%.
* The period-1 values match the committed file to the digits shown.
* With the daily losses averaged over the five seed sets, the picture is the same (robustness, `periods.csv`):
  * at 15 minutes, E changes by +7.07 and G by +1.03;
  * at 30 minutes, E changes by +8.76 and G by +2.06 (p 0.138).
  * B has no seeds, so it is unchanged.

### 4.2 Which group's contribution changed? (the interaction test)

Source: `tests.csv` rows `P2_*`, `verdict.csv`, `periods.csv`, `seed_repeats.csv` (part "2"). Contribution in
points of the period's mcsGARCH QLIKE; change = period 2 minus period 1; two-sided p; Holm over the 20 primary
rows. "Seed sets agreeing" = seed sets whose change has the headline's sign.

| Group | Horizon | Period 1 | Period 2 | Change | p | Holm p | Seed sets agreeing | Seed SD | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| realized | 15 min | +1.83 | +2.09 | +0.26 | 0.669 | 1.000 | 5 of 5 | 0.19 | not supported (within seed noise) |
| realized | 30 min | +2.97 | +2.86 | -0.11 | 0.930 | 1.000 | 2 of 5 | 0.64 | not supported (within seed noise) |
| time of day | 15 min | +1.68 | +1.15 | -0.52 | 0.264 | 1.000 | 5 of 5 | 0.27 | not supported (within seed noise) |
| time of day | 30 min | +1.30 | +2.29 | +0.99 | 0.209 | 1.000 | 5 of 5 | 0.08 | not supported |
| implied | 15 min | +0.46 | -0.08 | -0.54 | 0.499 | 1.000 | 4 of 5 | 0.38 | not supported (within seed noise) |
| **implied** | **30 min** | **+1.14** | **-1.16** | **-2.30** | **0.003** | **0.051** | **5 of 5** | 0.37 | **not supported** (just above the line) |
| VIX | 15 min | -0.07 | -0.25 | -0.18 | 0.421 | 1.000 | 3 of 5 | 0.26 | not supported (within seed noise) |
| VIX | 30 min | -0.04 | -0.41 | -0.37 | 0.393 | 1.000 | 2 of 5 | 0.40 | not supported (within seed noise) |
| calendar | 15 min | -0.13 | -0.56 | -0.43 | 0.276 | 1.000 | 5 of 5 | 0.19 | not supported |
| calendar | 30 min | +0.04 | -0.95 | -0.99 | 0.041 | 0.618 | 5 of 5 | 0.30 | not supported |

**The implied group at 30 minutes in detail.**

* **Seed sets.** The change is -2.30 (headline), -1.36, -1.45, -1.81 and -1.67 points in the five seed sets,
  mean -1.72 (`seed_repeats.csv`). The headline seed set gives the largest of the five.
* **Robustness rows** (`tests.csv`, each family Holm-adjusted on its own):

  | Check | Change | p | Holm p |
  |---|---|---|---|
  | R5, day-block bootstrap instead of the delta method | -2.30 | below 0.001 | 0.009 |
  | R7, period 2 cut at the closing auction (417 days) | -2.12 | 0.009 | 0.085 |
  | R8, without the 42 EWMA-fallback days (270 days in period 1) | -2.49 | 0.003 | 0.030 |
  | R9, daily losses averaged over the five seed sets | -1.72 | 0.014 | 0.136 |

* **Exploratory halves of the group** (E3, unadjusted): the change is -1.01 points for the level half (p 0.030)
  and -1.03 points for the shape half (p 0.039). Both halves fell, so neither alone accounts for the drop.
* **At 15 minutes** the same group's change is -0.54 (p 0.499); seed 104 has the other sign.

**Combined variants** (secondary family S3b, Holm over 4 rows):

| Variant | Horizon | Period 1 | Period 2 | Change | p | Holm p | Seed sets agreeing |
|---|---|---|---|---|---|---|---|
| `mcsinfo` (everything mcsGARCH lacks) | 15 min | +0.68 | -0.76 | -1.44 | 0.012 | 0.036 | 5 of 5 |
| `mcsinfo` | 30 min | +1.22 | -1.31 | -2.52 | 0.001 | 0.006 | 5 of 5 |
| `optionmkt` (implied and VIX) | 15 min | +0.73 | +0.53 | -0.21 | 0.833 | 0.833 | 5 of 5 |
| `optionmkt` | 30 min | +1.01 | -0.71 | -1.73 | 0.038 | 0.076 | 5 of 5 |

Before the cut, the information mcsGARCH lacks (option market, VIX, calendar) was worth about +0.7 and +1.2
points to the hybrid. After the cut it cost about 0.8 and 1.3 points. All five seed sets agree.

### 4.3 The three regimes (descriptive, no test)

Source: `regimes.csv` (headline seed set). Points of the regime's mcsGARCH QLIKE, except the "% of `mcsgarch_iv`"
column.

| Horizon | Regime | Days | G | B | E | E in % of `mcsgarch_iv` | realized | time of day | implied | calendar | `mcsinfo` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 min | multi-weekly | 312 | 5.19 | -4.98 | 0.21 | -0.22 | 1.83 | 1.68 | 0.46 | -0.13 | 0.68 |
| 15 min | Thursday weekly | 192 | 5.16 | 2.31 | 7.47 | -7.30 | 1.30 | 1.51 | -1.23 | -0.09 | -1.49 |
| 15 min | Tuesday weekly | 288 | 6.83 | 0.14 | 6.97 | -6.96 | 2.67 | 0.90 | 0.77 | -0.90 | -0.23 |
| 30 min | multi-weekly | 312 | 6.70 | -6.06 | 0.64 | -0.68 | 2.97 | 1.30 | 1.14 | 0.04 | 1.22 |
| 30 min | Thursday weekly | 192 | 6.42 | 1.99 | 8.41 | -8.25 | 1.59 | 2.08 | -1.19 | -0.15 | -0.87 |
| 30 min | Tuesday weekly | 288 | 9.73 | -0.49 | 9.24 | -9.29 | 3.92 | 2.46 | -1.13 | -1.63 | -1.68 |

The G and B columns equal the committed scoreboards' regime margins (Prior knowledge in the plan), which is a
check that the decomposition is computed on the same losses. At 30 minutes the implied group is negative in both
later regimes. At 15 minutes it is negative only in the Thursday regime.

## 5. Other secondary results (REAL DATA)

**S4, last hour** (Nifty, headline seed only; Holm over 5 rows; the hybrid's gain is 6.36 points;
`tests.csv` rows `S4_*`, `shares.csv`):

| Group | Contribution | Share [95% CI] | p | Holm p |
|---|---|---|---|---|
| realized | +4.38 | 69% [42%, 119%] | below 0.001 | below 0.001 |
| time of day | +2.21 | 35% [-18%, 103%] | 0.129 | 0.517 |
| implied | +0.39 | 6% [-22%, 29%] | 0.303 | 0.693 |
| VIX | +0.38 | 6% [-12%, 23%] | 0.231 | 0.693 |
| calendar | +0.09 | 1% [-11%, 19%] | 0.419 | 0.693 |

**S5 Bank Nifty** (660 test days to 22 April 2026; hybrid gain 3.92 and 5.03 points) and **S6 Fin Nifty** (604
test days to 27 January 2026; gain 4.90 and 6.09 points). Headline seed only; Holm over 10 rows per index;
`tests.csv` rows `S5_*`, `S6_*`, `shares.csv`, `contributions.csv`:

| Index | Group | 15 min: contribution (share), Holm p | 30 min: contribution (share), Holm p |
|---|---|---|---|
| Bank Nifty | realized | +1.58 (40%), below 0.001 | +1.79 (36%), below 0.001 |
| Bank Nifty | time of day | +0.88 (23%), below 0.001 | +0.95 (19%), below 0.001 |
| Bank Nifty | implied | -0.21 (-5%), 1.000 | -0.48 (-9%), 1.000 |
| Bank Nifty | VIX | -0.06 (-2%), 1.000 | -0.24 (-5%), 1.000 |
| Bank Nifty | calendar | -0.02 (-1%), 1.000 | +0.15 (3%), 1.000 |
| Fin Nifty | realized | +1.94 (40%), below 0.001 | +1.95 (32%), below 0.001 |
| Fin Nifty | time of day | +1.62 (33%), below 0.001 | +1.57 (26%), below 0.001 |
| Fin Nifty | implied | +0.20 (4%), 0.911 | +0.17 (3%), 0.911 |
| Fin Nifty | VIX | +0.15 (3%), 0.156 | +0.09 (1%), 0.911 |
| Fin Nifty | calendar | +0.20 (4%), 0.035 | +0.06 (1%), 0.911 |

* **The same two groups lead on all three indices.** The realized history comes first on every index at both
  horizons, as expected.
* **Ranking agreement.** The Spearman rank correlation of the five contributions with Nifty's is 0.6 and 0.7
  for Bank Nifty and 0.9 and 0.9 for Fin Nifty (`other_index_ranking.csv`, descriptive).
* **One small exception.** On Fin Nifty at 15 minutes the calendar group helps a little (+0.20 points, 4% of the
  gain, Holm 0.035). The other way round from Nifty, and only at one horizon on one index.

## 6. Robustness and placebo (REAL DATA)

**Part 1 robustness** (Holm within each family; `tests.csv` rows `R1_*`, `R2_*`, `R3_*`):

| Check | realized (15 / 30 min) | time of day (15 / 30 min) | implied, VIX, calendar |
|---|---|---|---|
| R1: daily losses averaged over five seed sets | +1.98 / +3.25, Holm below 0.001 | +1.33 / +2.24, Holm below 0.001 | none significant (smallest raw p 0.381) |
| R2: without the 09:15 forecast | +1.96 / +2.16, Holm below 0.001 | +1.33 / +1.92, Holm below 0.001 / 0.002 | none significant (smallest raw p 0.310) |
| R3: the 729 days before the closing auction | +1.84 / +2.81, Holm below 0.001 | +1.50 / +1.37, Holm below 0.001 | none significant (smallest raw p 0.571) |

**Part 2 robustness** (`tests.csv` rows `R5_*` to `R9_*`). Apart from the implied group at 30 minutes (Section
4.2), the only rows with a raw p below 0.05 are the calendar group at 30 minutes in R5 (bootstrap, p 0.036,
Holm 0.321) and in R8 (p 0.034, Holm 0.309), and the implied group at 15 minutes in R7 (-1.26 points, p 0.027,
Holm 0.244). R6 repeats S1 and S2 on the R7 and R8 subsets:

* the edge's change is +7.74 and +7.74 points before the closing auction (p 0.031 and 0.021), and +7.27 and
  +8.29 points without the EWMA days (p 0.035 and 0.006);
* the benchmark's change is +7.18 / +7.46 and +6.61 / +7.29;
* the hybrid's change is +0.56 / +0.28 and +0.65 / +1.01 (all p 0.5 or more);
* after Holm over the 12 R6 rows none is below 0.05 (smallest 0.067).

**Placebo** (Holm over 8 rows each; `tests.csv` rows `PL1_*`, `PL2_*`). The full hybrid with seed set 7 is
compared with the same hybrid with seed set 101, 102, 103 or 104: "removing nothing".

* **PL1 (Part 1's test on seed-only differences):** no row is significant. The smallest raw p is 0.056 (seed
  101, 15 minutes, +0.16 points), Holm 0.451. Part 1's test is not fooled by seed noise.
* **PL2 (Part 2's period test on seed-only differences):**
  * At 30 minutes, the seed-only change is -0.79 (p 0.081), -0.85 (p 0.037), -0.31 (p 0.341) and -0.77 points
    (p below 0.001, **Holm 0.002**) for seeds 101 to 104.
  * At 15 minutes all four are non-significant (p 0.083 to 0.726).
  * **So the period test can call a pure seed difference significant** (1 of 8 after Holm, 2 of 8 raw). Its
    p-values do not include the uncertainty from LightGBM's seeds. This is why the registered rule also asks
    all five seed sets to agree, and it is why the implied group's p of 0.003 is weaker evidence than it looks.
  * The implied group's change (-2.30; seed mean -1.72) is larger than any of the eight placebo changes
    (largest in size -0.85).

## 7. Exploratory results (REAL DATA, no adjustment)

* **E1, halves of the implied group, all days:**
  * level half: -0.18 points at 15 minutes and -0.15 at 30 minutes;
  * shape half: -0.03 and -0.39;
  * all have p of 0.545 or more.
* **E2, the 169 expiry days:**
  * the hybrid's gain on these days is 6.64 and 6.24 points;
  * realized: +1.99 (p 0.004) and +1.80 (p 0.019);
  * time of day: +1.64 (p below 0.001) and +1.04 (p 0.044);
  * implied: +0.37 and +0.23 (p 0.325 and 0.389);
  * VIX: -0.18 and -0.27;
  * calendar: -0.37 and -0.94.
  * Even on expiry days, the option market adds nothing measurable.
* **E3, period change of the implied halves:** +0.11 and -0.14 at 15 minutes (p 0.792 and 0.727); -1.01 and
  -1.03 at 30 minutes (p 0.030 and 0.039).
* **E4, last hour:**
  * `optionmkt` +0.36 (p 0.337);
  * `mcsinfo` +1.29 (p 0.107);
  * `implied_level` +0.70 (p 0.131);
  * `implied_shape` +0.07 (p 0.470).
* **E5, last-hour period changes:**
  * edge E +3.41 (p 0.560), hybrid G -4.73 (p 0.259), benchmark B +8.14 (p 0.302);
  * groups: VIX +2.13 (p 0.011), time of day +3.77 (p 0.048), the other three p 0.379 or more.
  * With a single seed set and P99's last-hour seed SD of 0.244 points for the margin, these two raw p-values
    are not interpreted.

## 8. What we expected, and what happened

| Row | Registered expectation | Outcome |
|---|---|---|
| P1 realized | supported at both horizons; the largest; more than a third of the gain | **as expected**: supported at both, the largest, 34% and 38% |
| P1 time of day | supported at both horizons, 0.5 to 2 points | **as expected**: supported at both, 1.34 and 1.92 points |
| P1 implied | not supported, or below 0.5 points | **as expected**: +0.11 and -0.29, not supported |
| P1 VIX | not supported | **as expected** |
| P1 calendar | not supported | **as expected**, and the point estimate is negative at both horizons |
| P2 implied | smaller since the cut (D < 0); contradicted if D > 0 and supported | **direction as expected at both horizons** (-0.54 and -2.30) but **not supported** (Holm 0.051 at 30 minutes); not contradicted |
| P2 other groups | no direction stated | none supported |
| S1 | the edge over `mcsgarch_iv` larger in period 2 | **as expected**, Holm 0.032 and 0.009 (a re-test on the same data, not a new confirmation) |
| S2 | the benchmark's change large and significant, the hybrid's small, B's share above 0.5; contradicted if its interval lies wholly below 0.5 | **partly**: the benchmark's change is large (6.04 and 6.70) but its Holm p is 0.135; the hybrid's is small and not significant; share 0.87 and 0.82, intervals not below 0.5 (not contradicted) |
| S3 | (b) yes; (a) and (c) open | (b) yes, p below 0.001 at both; (a) and (c) not significant |
| S5, S6 | the realized group first on every index | **as expected** |

## 9. Checks

* **Reproduction of the committed forecasts** (`reproduction_check.csv`). The full hybrid with seed 7, refitted
  through the new code path, is identical to the last bit to the committed `hybrid_base` forecasts, together
  with `y`, `mcsgarch` and `mcsgarch_iv`, on every row:
  * Nifty: 19,800 rows at 15 minutes, 19,008 at 30 minutes and 792 in the last hour;
  * Bank Nifty: 16,500 and 15,840 rows;
  * Fin Nifty: 15,100 and 14,496 rows.
* **Reproduction of P99.** With seeds 101 to 104, the full hybrid's daily QLIKE equals P99's committed
  `results/seed_bootstrap/daily_losses_<target>.csv` on all 792 days, for each seed set and both horizons, and
  so does mcsGARCH's (largest relative difference 0). This check needed a bug fix after the first analysis;
  see Deviations.
* **Inputs.** Every raw file each run read matches `data/MANIFEST.csv` (`run_info.json`,
  `input_sha256.<index>.manifest_check`): 1,891 files for Nifty, 1,891 for Bank Nifty and 1,711 for Fin Nifty,
  none differing.
* **Period labels.** The date split (20 November 2024) equals the pipeline's regime label on every day (checked
  inside `analyze`, which refuses to run otherwise).
* **Same benchmarks in every file.** `y`, `mcsgarch` and `mcsgarch_iv` are identical in every work file of a
  target (checked inside `analyze`).
* **Tests.** `tests/test_p106_gain_decomposition.py` (SYNTHETIC only) has 16 tests: the 15 registered ones and
  one added for the bug fix. The full test suite passes (run before the phase-2 commit).

## 10. Run time

* **Forecasts** (stage 1), on 2 worker processes:
  * from 06:02 to 07:47 UTC on 3 November 2026 (`run_info.json`, `stage1_wall_clock`, from the work files' times);
  * Nifty took 87.2 minutes (22 walk-forwards fitting 94 variants);
  * Bank Nifty took 9.4 minutes and Fin Nifty 9.1 minutes (4 walk-forwards and 12 variants each;
    `run_stdout_*.txt`);
  * the jobs' own times add up to 11,703.5 seconds (`timing.csv`).
* **Analysis** (stage 2): 30 seconds (`run_info.json`, `seconds_stage2`; the first analysis run took 32 s).
* The plan expected 1.5 to 2 hours in all. The run took about 1 hour 45 minutes.

## 11. Deviations from the plan

1. **Bug in the P99 reproduction check, fixed after the first analysis (outcomes had been seen).**
   * **What went wrong.** The registered code compared the seed-101-to-104 daily losses with P99's file by
     reindexing a day frame indexed by date text on P99's datetime index. So it matched no day: the first
     analysis wrote `rows_matched` 0 for those eight rows of `reproduction_check.csv`.
   * **The fix.** The comparison now aligns both on the date written as text (new function `p99_daily_check`,
     new synthetic test `test_p99_daily_check_aligns_text_and_datetime_dates`), and `analyze` was run again.
   * **Effect.** The check now reports 792 of 792 days identical for every seed set and horizon, and a direct
     comparison of the two files, made before the fix, found the same. Every other output file of the second
     analysis is byte for byte the same as the first (checked with `cmp`): `tests.csv`, `verdict.csv`, `contributions.csv`,
     `shares.csv`, `periods.csv`, `regimes.csv`, `seed_repeats.csv`, `other_index_ranking.csv`, the daily losses
     and `timing.csv`. Only `reproduction_check.csv`, `run_info.json` and `analyze_stdout.txt` changed. No
     estimate or p-value changed.
2. **`run_info.json` gained a field after the run**: `stage1_wall_clock` (start and end of each index's stage 1,
   from file times), added so that the run time is in the file as the brief asks. It changes no number.
3. **`analysis_commit` in `run_info.json` reads 71bb3b6** (the plan commit), because `analyze` records the
   commit that was checked out. The analysis ran on that commit plus the two changes above, which are in the
   phase-2 commit.
4. **The `check` stage was also run once during the run** (06:30 UTC, after the first headline-seed jobs had
   finished), on the real work folder. It compares the full hybrid's forecasts with the committed ones and
   computes no loss; all were identical. The plan registers this stage but is silent on when it is run.
5. **Commands.** The plan's commands set `PYTHONPATH=src`; the run used the absolute path of the same `src`
   folder (a sandbox rule). Same code.
6. **Two typing errors in the plan's Prior knowledge table** (no effect on the design). The plan gives the
   committed 30-minute edge over `mcsgarch_iv` as -0.70% (multi-weekly) and -8.21% (Thursday weekly). The
   committed `results/referee_forecast/gw_group_margins.csv` says -0.68% and -8.25%, and this study reproduces
   -0.68% and -8.25% (Sections 4.1 and 4.3).
7. **Things the plan was silent on.**
   * The calendar group's *negative* contribution is reported from the registered share interval only. No test
     in that direction was added, so tests.csv has no post hoc rows.
   * The R5 rows (bootstrap) have no test statistic, so their `statistic` column is empty.
   * The S3 (b) row's estimate is a margin in % of mcsGARCH's QLIKE (negative: the `mcsinfo` hybrid is better),
     and its statistic is the DM statistic of mcsGARCH's loss against the `mcsinfo` hybrid's (positive: the
     hybrid is better).
8. **Bank Nifty and Fin Nifty were run** (finished 07:47 UTC, before the 10:30 UTC cut-off), so nothing was
   dropped.

## 12. Caveats

1. **Leave-one-out measures what a group adds given all the others.**
   * The five shares add up to only about half the gain. The rest is carried jointly: a group that looks
     useless alone may matter if its stand-in is also removed.
   * The `optionmkt` and `mcsinfo` variants rule this out for the option market, VIX and calendar together.
   * They do not rule it out between the realized history and the time of day, which share mcsGARCH's own
     forecast in different ways.
   * A full attribution over all 32 subsets of groups was not attempted.
2. **The protocol is fixed.** Each variant keeps the headline's LightGBM settings. A hybrid with fewer inputs
   might do better if tuned; this measures the loss under the headline's protocol.
3. **Part 2 uses the same days that motivated it.**
   * The edge's period pattern and its split into benchmark and hybrid were visible in committed files before
     this study.
   * The direction for the implied group was chosen after seeing that a related block helped only before the
     cut. The group-level tests are new, but they use the same 792 days.
4. **A period is a stretch of time.** A change between the periods may come from SEBI's cut, from the longer
   training window available later, or from anything else that changed in 2025 and 2026. The design cannot
   separate these.
5. **The period test ignores seed noise.**
   * The placebo shows a seed-only change can get p below 0.001.
   * Five seed sets only show whether a sign is stable; they cannot estimate the seed noise of a period change
     precisely.
   * The last hour and the other two indices use one seed set.
6. **The benchmark's share of the change in the edge is imprecise at 30 minutes** (interval 0.20 to 1.18), and
   the benchmark's change on its own does not survive Holm (0.135).
7. **The benchmark is our implementation of mcsGARCH**, with an EWMA fallback in two refits. R8 leaves those 42
   days out and changes nothing material.
8. **Grouping choices are fixed in the plan.** For example, mcsGARCH's own forecast sits in the realized group,
   and the implied-minus-realized spread in the implied group. Other choices are defensible.

## 13. What this changes in the project's claims

* **The hybrid's gain can now be attributed.**
  * About a third comes from recent realized volatility (34% and 38% at 15 and 30 minutes) and about a quarter
    from the time of day (23% and 25%), each with all five seed sets agreeing.
  * The same two groups lead on Bank Nifty and Fin Nifty, and the realized history carries most of the
    last-hour gain (69%).
* **The gain does not need the option market.**
  * On Nifty, removing the 16 option-implied inputs, VIX, or both together does not significantly raise the
    loss at 15 or 30 minutes or in the last hour. On Bank Nifty and Fin Nifty, removing the implied group or VIX
    does not either.
  * A hybrid that sees only mcsGARCH's own information gets the whole gain (6.06% and 8.01% better than
    mcsGARCH).
  * Any sentence that credits the option-implied features with the hybrid's gain should go. LightGBM's
    split-gain importance (30.0% and 32.7% for the implied group, per the plan's Prior knowledge, from
    `results/pipeline_nse_v2/feature_importance_<target>.csv`) does not show
    what the hybrid would lose without them, and should not be cited as evidence that they matter.
* **The November 2024 result should be worded as "the benchmark lost its edge", not "the hybrid got better".**
  * The hybrid's edge over the information-matched `mcsgarch_iv` appears after the cut mainly because
    `mcsgarch_iv` lost its advantage over mcsGARCH (about 82% to 87% of the change).
  * The hybrid's own gain over mcsGARCH was steady across the cut (no significant change).
* **No group-level interaction is established.**
  * The option-implied group's value at 30 minutes fell after the cut (from +1.14 to -1.16 points), but this
    misses the registered threshold (Holm 0.051). It should be called suggestive.
  * The broader block "everything mcsGARCH lacks" did fall significantly at both horizons, but that is a
    secondary result.
  * If reported, it should be said that this fall works against the hybrid's edge in period 2, not for it.
* **The calendar inputs do not help on Nifty** (point estimates negative at both horizons, share interval below
  zero). Dropping them, or using the `mcsinfo` hybrid, would be a new model chosen after seeing these data. It
  would need its own registered test on data not used here before it could become a headline.

## 14. Files

All in `results/p106_gain_decomposition/` (REAL DATA):

* `tests.csv`: every p-value (189 rows: 20 primary, 41 secondary, 82 robustness, 16 placebo, 30 exploratory),
  with the registered columns; `tests_detail.csv` adds the identifiers and sample details.
* `verdict.csv`: the 20 primary rows with seed agreement, seed SD and verdict.
* `contributions.csv`, `shares.csv`, `periods.csv`, `regimes.csv`, `seed_repeats.csv`,
  `other_index_ranking.csv`: the tables above.
* `daily_losses_nifty_<target>_s<seed>.csv`: each variant's daily QLIKE, from which every number can be
  recomputed.
* `reproduction_check.csv`, `timing.csv`, `run_info.json`: checks, timings, plan commit, seeds, input hashes and
  versions.
* `run_stdout_nifty.txt`, `run_stdout_banknifty.txt`, `run_stdout_finnifty.txt`, `analyze_stdout.txt`: logs.

The forecasts of every variant are in `data/processed/p106/` (git-ignored, about 240 MB). They can be remade with
the plan's commands in about 1 hour 45 minutes on 2 cores.
