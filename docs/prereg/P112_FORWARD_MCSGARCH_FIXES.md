# P112: forward test of the five fixes to mcsGARCH (registered plan)

**Status.** Written on Tuesday 3 November 2026, during the Nifty session, by an AI agent (Claude) acting as the
project lead. No data from Wednesday 4 November 2026 or later existed when it was written. The registration is
**the commit that adds this file**; its git time stamp proves when the plan was fixed. The analysis must be run
with the code at that commit ("the pinned commit"). Any later change is a deviation (section 8).

## 1. Why

Study P109 (`docs/P109_MCSGARCH_EXTENSIONS.md`, plan `ceec2ec`, results `7f0b5c7`) wrote down five fixes to the
paper's mcsGARCH model before computing any result, and found that the five together (the **combination**) beat
mcsGARCH by 6.83 and 8.25 points of mcsGARCH's QLIKE at 15 and 30 minutes on 792 Nifty test days (one-sided
Diebold-Mariano t 5.61 and 5.01), as much as the LightGBM hybrid. But the five fixes were chosen after study
P106 had been run on the same days, so P109 explains the hybrid's gain; it does not show that the combination
forecasts better on data it has never touched. Only days that have not happened yet can show that. This file
fixes, before they exist, the one test that will be run on them, when, and what each result will mean.

It is separate from the forward test of `docs/prereg/FORWARD_TEST.md` (H1 and H2), which it does not change.

## 2. Models

* **mcsGARCH**: the committed headline implementation (`src/niftygex/volatility.py`, as used for
  `results/pipeline_nse_v2/`), with its daily AR(1)-EGARCH(1,1) level, diurnal curves and intraday Student-t
  GARCH(1,1).
* **combination**: P109's `combination` variant exactly as coded at commit `7f0b5c7` (`McsGarchExt` in
  `src/niftygex/volatility.py`, run by `experiments/p109_mcsgarch_extensions.py`): X1 (log-HAR daily level),
  X2 (overnight fade), X3 (today's running level), X4 (leverage) fitted jointly by maximum likelihood, then X5
  (one calibration factor per anchor minute from the training days).
* **hybrid** (secondary only): `hybrid_base` of the v2 pipeline (`scripts/run_pipeline.py --source nse
  --n-seeds 5`, same settings as `results/pipeline_nse_v2/`).

No parameter, feature, rule, window length, refit interval, seed or calibration rule may be changed. Each
model is refitted every 21 trading days on all data before the refit day, exactly as in P109's `run` stage,
with the walk-forward simply continued forward in time.

## 3. Data and sample

* **Index and targets**: Nifty 50; the realized variance of the next 15 and the next 30 minutes, at the same
  anchor minutes as P109 (`target_specs` in `experiments/p109_mcsgarch_extensions.py`). The last-hour target is
  not tested (P109 found no gain there).
* **Test days**: every Nifty trading day from **Wednesday 4 November 2026** onward whose one-minute bars pass the
  project's validation (`scripts/validate_minute_bars.py`: daily high and low within 0.05% of NSE's official
  values). 3 November 2026 is excluded because this plan was written during its session.
* **Stopping point**: the 250th valid test day, or 31 December 2027, whichever comes first. If fewer than 120
  valid test days exist at that date, the test is reported as underpowered and descriptive only (no verdict).
* **Minute bars**: the pinned public source used so far (`data/raw/minute_bars_source.json`), updated to the
  stopping point. If that source stops, any other source of NSE Nifty one-minute bars may be used if it passes
  the same validation on every test day; the switch is reported.
* All forward days are closing-auction days (from 3 August 2026); they are treated exactly as the 63 auction
  days already inside P109's sample.

## 4. Hypothesis and test

**Primary (H1)**: on the forward test days, the combination's mean daily QLIKE is lower than mcsGARCH's, at
15 minutes and at 30 minutes.

* **Test**: for each horizon, the one-sided Diebold-Mariano test on daily mean QLIKE used in P109 (Newey-West
  variance, Harvey-Leybourne-Newbold factor, Student t with n - 1 degrees of freedom), direction "combination
  has lower loss". Estimate: the gain in points of mcsGARCH's QLIKE, 100 x (L_mcsGARCH - L_combination) /
  L_mcsGARCH.
* **Holm over the two horizons** at 5%: the smaller p must be at most 0.025, then the larger at most 0.05.
* **Verdict**:
  * **confirmed** at a horizon if its Holm-adjusted p is at most 0.05;
  * **not confirmed** otherwise;
  * **contradicted** at a horizon if the opposite one-sided p (mcsGARCH has lower loss) is at most 0.05.

**Secondary (outside Holm; cannot change the verdict)**:

* S1: the hybrid against the combination, two-sided Diebold-Mariano at each horizon (P109 could not tell
  them apart).
* S2: the share of the hybrid's gain that the combination recovers, with a day-level bootstrap interval as in
  P109's `recovered_share`.
* S3: the primary test with the forecasts made at 09:15 left out (P109 found that 37% to 39% of the combination's
  gain is at the open).

## 5. Power (rough, stated before the data)

If the forward days behaved like P109's 792 days, the t statistic would scale with the square root of the
number of days: about 5.61 x sqrt(250 / 792), roughly 3.2, at 15 minutes and 2.8 at 30 minutes, which would
confirm both. If the true gain were half of P109's, the expected t would be about 1.6 and 1.4, and the chance of
confirming would be roughly one half or less. A "not confirmed" is therefore weak evidence against a gain half
the size of P109's.

## 6. When it runs

Once, at the stopping point. **No interim look** at the forward losses of these models for this purpose. If
the losses are computed earlier for any other reason (for example a demo), that is reported, and the test
still runs only at the stopping point.

## 7. What each result will mean

* **Confirmed at both horizons**: the five fixes improve the paper's model on data they have never touched;
  the combination can then be offered to the paper's authors as a better benchmark.
* **Confirmed at one horizon only**: report the horizon; no general claim.
* **Not confirmed**: the P109 result stays an in-sample explanation of the hybrid's gain, nothing more.
* **Contradicted**: the fixes were fitted to the 792 days; withdraw any suggestion that they improve mcsGARCH.

## 8. Deviations

Any change to the code at the pinned commit, the models, the data rules or the test, and any look at the
forward losses before the stopping point, is reported as a deviation with its reason and date, next to the
result.
