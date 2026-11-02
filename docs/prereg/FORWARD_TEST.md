# Forward test: pre-registration for Nifty data from 3 November 2026

**Status.** Written on 2 November 2026, after the close of the last in-sample day (Monday 2 November 2026)
and before any market data for 3 November 2026 or later existed. The pre-registration is **the commit that
adds this file**; its git time stamp proves when the plan was fixed. **The analysis must be run with the code
at that commit** ("the pinned commit"). Any later change is a deviation (section 9).

## 1. Why, and words used

Every claim so far is in sample or pseudo out of sample (the T4 split of FALSIFICATION was chosen after the
full-sample result was known). The only unseen data are days that have not happened yet. This file fixes,
before they exist, which tests will run on them, when, and what each result will mean.

* **p-value**: the chance of a result at least this extreme if there were no effect. A **one-sided** test asks
  only about the predicted direction; the **opposite-direction p** asks about the other one.
* **Holm** (two tests): the smaller p must be at most 0.025, then the larger at most 0.05, which keeps the
  chance of any false "confirm" at 5%. **HAC errors, 5 lags**: standard errors allowing nearby days to be related.
* **Pro GEX**: dealer gamma exposure signed by NSE's "Pro" participant positions (`docs/FALSIFICATION.md`).
* **Power**: the chance the test confirms a real effect of a given size. **Stopping point**: when a test's data
  are complete; each test is run once, then, never before.

## 2. What the closing auction changes

From 3 August 2026 NSE sets the official close in an auction. On all 63 auction days in the sample (checked
for this file on `data/raw/nifty_1min.csv`), bars up to 15:14 trade normally (none from 14:15 to 15:14 has high
equal to low); the 15:15 bar is a single price and every close from 15:16 to 15:27 equals it, so the index
**freezes at the 15:15 close**; the 15:29 close is the auction price and equals NSE's official close. Every
forward-test day will be like this. R5's "last 30 minutes" (14:59 to 15:29) now holds 16 minutes of trading,
13 frozen minutes and the auction jump; R2's "last hour" (bars 14:30 to 15:29) holds 13 frozen minutes, which
add exactly zero variance, and one auction return.

## 3. The hypotheses

Two confirmatory hypotheses with Holm across them. **There is no H3** (section 3.4).

### 3.1 H1: the end-of-day reversal after negative Pro GEX (F2)

**Claim**: in R5's regression (`experiments/momentum_gex.py`, specification "binary", all days)
`r_last = a + b r_rod + c (r_rod x neg) + d neg + error`, the coefficient c (`r_rod_x_neg`) is **negative**.

* Sample: every trading day t from 3 November 2026 to the stopping day on which r_rod, r_last and neg exist.
* **neg** = 1 if Pro GEX at the previous close is below 0, computed exactly as in `momentum_gex.py`
  (`net_gex_participant` of day t-1 over the 20-day rolling median of gross GEX, at least 5 days); 0 otherwise;
  missing if GEX is missing.
* **r_rod** = log(14:59 bar close on day t / 15:29 bar close on day t-1) x 10,000, in basis points (bp).
* **Primary r_last** = log(15:29 bar close / 14:59 bar close) x 10,000, the formula used so far; on auction
  days it ends at the auction price, which equals the official close.
* **Test**: OLS with HAC errors, 5 lags (`momentum_gex.fit`); z = c / SE; one-sided p = Phi(z), with Phi the
  normal distribution function (statsmodels' HAC p-values are normal, as in T4); opposite p = 1 - Phi(z).
* Inputs dated t-1 (GEX, the previous close) are known before day t opens, so 3 November 2026 uses those of
  2 November. No outcome from before 3 November 2026 enters.

**Secondary H1b** (outside Holm; cannot change a verdict): the same with r_last = log(15:15 bar close /
14:59 bar close) x 10,000, which ends where the index freezes: all continuous trading, no auction jump.

### 3.2 H2: the Tuesday expiry-day last-hour premium

**Claim**: on Nifty's weekly expiry day the last hour of continuous trading is more volatile than the same hour
on the other days of its week: R2's week-relative log realized variance is **positive**.

* **Realized variance (RV)** of a day: the sum of the 60 squared one-minute log returns from the 14:15 bar
  close to the 15:15 bar close; missing if any of these 61 closes is missing.
* **Week-relative value** (`real_day_level.week_relative`): for each calendar week with exactly one Nifty
  expiry day, log RV on the expiry day minus the mean log RV of the week's other trading days. A week counts
  once its Friday has passed, if the expiry day and at least two other days of the week lie inside the forward
  window with a valid RV. Days outside the window are never used (the week of 3 November 2026 compares Tuesday
  with 4, 5 and 6 November, not with Monday 2 November).
* **Counting weeks**: only weeks whose expiry falls on Tuesday, or on the last trading day before a Tuesday
  holiday.
* **Test**: Wilcoxon signed-rank, `scipy.stats.wilcoxon(x, alternative="greater")` with default settings, as
  in R2 but one-sided; the opposite p uses `alternative="less"`.

**Why 14:15 to 15:15.** It holds only continuous trading: no frozen minutes (which add exactly zero) and no
auction jump (a price set by a different mechanism, whose single large return would dominate the sum). It
keeps 60 returns, like R2's last hour, and ends at the last traded level. The cost: it drops the minutes
closest to settlement, so the premium may be smaller than R2's (one reason section 5 also uses half of it).

**Secondary H2b** (outside Holm): the same with Sensex expiry days that are not Nifty expiry days left out of
the baseline, as in `docs/SENSEX_CONFOUND.md` (Table 4), using the rule in
`data/external/sensex_expiry_calendar.csv` at the pinned commit (Thursday from 1 September 2025, previous
trading day on a holiday; reported, not verified, because BSE's servers are blocked here).

### 3.3 Descriptive only (no decision)

H1: the momentum-only slope. H2: the premium in percent with a bootstrap 95% interval (as R2), the share of
positive weeks, a t-test of the mean, and the same weeks with R2's original last hour (bars 14:30 to 15:29,
auction included), to show whether the premium moved into the auction. Data checks on bars and the auction.

### 3.4 No H3: to be pre-registered after the pipeline is frozen

The forecasting findings depend on the pipeline (`scripts/run_pipeline.py`, `src/niftygex/forecast.py`),
which is being changed now, so they wait for a later pre-registration: (a) the hybrid beats mcsGARCH at 15 and
30 minutes (finding 2); (b) no model beats mcsGARCH for the last hour (finding 3); (c) GEX features add no
out-of-sample value (finding 4). R3's expiry-day flip is left out: a third test would raise H1's bar.

## 4. Decision rules

At a hypothesis's stopping point, with p the one-sided p in the predicted direction and q the opposite p:

* **Confirm**: Holm-adjusted p at most 0.05 (Holm over H1 and H2, always two tests; a test that is not
  evaluable counts as p = 1).
* **Contradict**: q below 0.05 (significant in the opposite direction).
* **Fail to confirm**: neither.
* **Not evaluable**: H1 has fewer than 30 days with neg = 1 or 30 with neg = 0, or H2 fewer than 20 counting
  weeks. No claim either way.

The stopping points differ, so the first test to stop is classified at once where Holm allows: p at most
0.025 confirms (Holm cannot undo it); p above 0.05 fails; q below 0.05 contradicts; p between 0.025 and 0.05 is
**pending**, and confirms only if the other test ends with p at most 0.025. H1b and H2b are read at 0.05
without Holm and labelled secondary. `scripts/forward_test.py` applies all of this mechanically.

## 5. Stopping rules and power

**H1 stops** on the day of the **130th** sample day with neg = 1 after 2 November 2026, or on **31 December
2029**, whichever is first. **H2 stops** at the **100th counting week**, or with the last counting week whose
Friday is on or before **31 December 2028**, or with the last counting week before the first Nifty weekly
expiry that breaks the Tuesday rule, whichever is first.

**Power** of a one-sided test at level alpha for a true effect delta with standard error SE is about
Phi(|delta| / SE - z), z = 1.645 (alpha 0.05, the level once the other test has confirmed) or 1.960 (alpha
0.025, the Holm worst case). "Half" is the in-sample estimate halved: if the truth were half, significant
in-sample estimates would on average be 1.26 to 1.45 times too large for R5 and 1.21 to 1.27 for the Tuesday
last hour (`docs/POWER_REAL.md`, section 5). `python scripts/forward_test.py --power` prints every number below
from the files named.

**H1.** In sample c = -0.068196, HAC SE 0.018433, 929 days: 463 with negative and 466 with non-negative Pro GEX
(`results/momentum_gex/regressions.csv`, `gamma_states.csv`). c is a difference of two slopes, so
SE = k x sqrt(1/n_neg + 1/n_other) with k = 0.018433 / sqrt(1/463 + 1/466) = 0.2809. On 2025 onward (98 and 354
days) this gives 0.0321 against an actual 0.0293, and before the auction (90 and 299) 0.0338 against 0.0295
(`results/falsification/t4_preregistered_f2.csv`): slightly conservative. Negative Pro GEX is now rarer: 98 of
452 days from 2025 (21.7%, T4 file), 49 of 288 since Tuesday expiry began on 1 September 2025 (17.0%,
`results/pipeline_nse_v2/daily_gex_panel.csv`), 8 of 63 auction days (12.7%, T4 file). Trading days were 245,
245 and 247 in 2023 to 2025 (`daily_market.csv`), so 245 a year is assumed, plus 41 for the rest of 2026 (the
count for 3 November to 31 December 2025): about 286 days to 31 December 2027 and 776 to 31 December 2029.

| H1 scenario | Negative days | Other days | SE | Power at -0.0682 (alpha 0.05 / 0.025) | At half, -0.0341 |
| --- | --- | --- | --- | --- | --- |
| Stops at 130, rate 17.0% (about December 2029) | 130 | 634 | 0.0270 | 81% / 71% | 35% / 24% |
| Stops at 130, rate 21.7% (about April 2029) | 130 | 470 | 0.0278 | 79% / 69% | 34% / 23% |
| Cap of 31 Dec 2029 first, rate 12.7% | 98.5 | 677.5 | 0.0303 | 73% / 61% | 30% / 20% |
| Not chosen: stop on 31 Dec 2027, rate 17.0% | 48.7 | 237.3 | 0.0442 | 46% / 34% | 19% / 12% |
| Not chosen: stop at 100, rate 17.0% | 100 | 488 | 0.0308 | 71% / 60% | 29% / 20% |

First row: 130 x (1 - 0.170) / 0.170 = 634 other days; SE = 0.2809 x sqrt(1/130 + 1/634) = 0.0270;
0.0682 / 0.0270 = 2.52; Phi(2.52 - 1.645) = 0.81. 130 is the round count that gives about 80% power at the
in-sample estimate; a stop at the end of 2027 would give less than even odds of confirming an effect exactly
as large as estimated. The price is time: at recent rates the answer comes in 2029. At half the estimate H1
has about a one-in-three chance, and no feasible rule fixes that (it needs four times the negative days).

**H2.** In sample 0.356 log points (43% more variance) on 61 weeks, bootstrap 95% interval 0.196 to 0.511
(`results/real_day_level/r2_expiry_effect_by_regime.csv`). The simulated 80%-power detectable effect of this
signed-rank test is 0.2269 (residual scheme) or 0.2308 (wild), against a textbook 2.8 SE of 0.2246
(`results/power_real/headline.csv`). The less favourable gives an effective SE of 0.2308 / (1.960 + 0.842) =
0.0824 (plain SE from the interval: 0.0805), so SE(n) = 0.0824 x sqrt(61 / n). There were 52.2 counting weeks
a year in sample (`docs/POWER_REAL.md`), so 100 weeks take about 1.9 years (about October 2028).

| H2 case | Weeks | SE | Power at 0.356 (alpha 0.05 / 0.025) | At half, 0.178 |
| --- | --- | --- | --- | --- |
| Stops at 100 weeks (about October 2028) | 100 | 0.0643 | over 99% / over 99% | 87% / 79% |
| Expiry weekday changes after a year | 52 | 0.0892 | 99% / 98% | 64% / 51% |
| Minimum for a verdict | 20 | 0.1439 | 80% / 70% | 34% / 24% |

100 weeks gives about 80% power at half the estimate even at the Holm worst case (0.178 / 0.0643 = 2.77;
Phi(2.77 - 1.960) = 0.79). Caveat for both tables: the inputs come from the old definitions (R2's last hour was
14:30 to 15:29; R5's r_last crossed an auction on only 63 of 929 days); the new ones had not been computed on
past data when this section was written.

## 6. What each outcome would mean

**H1 confirmed**: the reversal after negative-Pro-GEX evenings holds on unseen days; F2 is a confirmed
statistical regularity, though still not explained (finding 8: it fits hedging theory only if the Pro sign is
backwards) and not tradable (`docs/F2_ECONOMICS.md`). **Fails to confirm**: F2 becomes "found in 2023 to 2026,
not confirmed later"; with about 80% power at the in-sample size, the true effect is likely smaller than
estimated, not necessarily zero. **Contradicted**: the last half hour continues the day's move after such
evenings; F2 is withdrawn. **H1b** only qualifies this: H1 without H1b means the reversal runs through the
auction price; H1b without H1 means the auction undoes it.

**H2 confirmed**: under the auction the expiry day's last hour of continuous trading is still more volatile
than the rest of its week; finding 6's Tuesday result replicates. **Fails to confirm**: the premium in 14:15 to
15:15 shrank (with 87% power at half the estimate, likely below half of R2's) or moved into the auction (the
14:30 to 15:29 numbers may hint, but support no claim). **Contradicted**: the expiry hour is calmer; R2's
Tuesday result is read as specific to its period. **H2b** supporting while H2 fails would suggest the Sensex
Thursday masks the premium; it cannot change the verdict.

**Not evaluable**: the question cannot be answered as specified in the auction era; reported with the counts.

## 7. No peeking

* No confirmatory, secondary or descriptive statistic of section 3 is computed on data after 2 November 2026
  before that hypothesis's stopping point, by `scripts/forward_test.py` or any other script
  (`momentum_gex.py`, `real_day_level.py`, `closing_auction_check.py`, `falsification.py`, `f2_economics.py`
  or new code).
* Allowed before then: **counts of days** only. `scripts/forward_test.py` with an `--end` before a stopping
  point prints, for that hypothesis, only the numbers of trading days, complete H1 days, negative-GEX days and
  counting weeks, plus data checks. The live dry run (`experiments/live_dry_run.py`) may keep scoring single
  days as a demonstration; those outputs are never pooled into an H1 or H2 statistic.
* When one hypothesis stops it is evaluated once; the other stays blind until its own stopping point.
* Any accidental look is reported in the write-up with its date and what was seen.

## 8. Data, commands and pinned code

**Data.** NSE's archive server: F&O bhavcopies (option chains, open interest, expiry calendar),
participant-wise open interest (the Pro sign), the daily index file (official Nifty 50 values, trading days).
Nifty 50 one-minute bars from `github.com/technovusin/nifty50-historical-data` (MIT), checked by the script
against NSE's daily high, low and open (5 bp, as R0); if over 2% of days fail, no test is run.

**Commands** (END = the stopping day or later). Files go to `data/live/forward`, which git ignores, so the
hashed `data/raw` never changes; starting on 1 September 2026 gives the GEX and close of 2 November 2026:

    python scripts/download_nse.py --start 2026-09-01 --end END --out data/live/forward
    python scripts/fetch_minute_bars.py --out data/live/forward --first-year 2026 --until END
    OMP_NUM_THREADS=1 python scripts/forward_test.py --start 2026-11-03 --end END --raw data/live/forward

The script builds the daily table and GEX panel with the pipeline's own loaders (no forecasting), refuses
other start dates except `--dry-run` on data up to 2 November 2026, applies the stopping rules, and writes
`results/forward_test/2026-11-03_END/` (`summary.md`, `summary.json`, and the day and week tables used).

**Pinned code.** Run from a checkout of the commit that adds this file (for example a separate git worktree).
`summary.json` records the commit it ran from and the SHA-256 of the script and of what it uses:
`experiments/momentum_gex.py` (`fit`, `RULES`), `experiments/real_day_level.py` (`week_relative`),
`experiments/sensex_confound.py` (Sensex rule), `scripts/run_pipeline.py` (`load_market`, `write_daily_market`),
`src/niftygex/` and the Sensex calendar. The dry run `results/forward_test/dryrun_2025-01-01_2026-07-31/` is
a **software check on past data, not the forward test**; it was run after this design was fixed and changed
nothing in it.

## 9. Deviations

Anything not described here is exploratory and labelled so. If NSE changes the auction, the definitions stay
and the script's auction counts record it; if NSE moves the weekly expiry, H2 stops (section 5) and H1
continues; if BSE moves its expiry, H2b keeps the pinned rule; a replacement minute-bar source must pass the bar
check. A bug fix is allowed only if the pinned code cannot compute what this file defines; the pinned result,
if it runs, is reported first.
