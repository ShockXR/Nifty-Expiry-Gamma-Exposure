# Does the forecasting result depend on LightGBM's random seeds? (P99 results)

**Verdict: the headline HOLDS.** We refitted the forecasting hybrid 20 more times, each time with 5 new random
seeds for LightGBM, on exactly the same data and settings. All 20 refits beat mcsGARCH at 15 and at 30 minutes
on all 792 test days, every one with a Diebold-Mariano p far below 0.05 (largest p 1.1e-8 at 15 minutes and
1.8e-13 at 30 minutes). The QLIKE gain ranges from 5.64% to 5.92% at 15 minutes (headline 5.80%) and from
7.76% to 8.11% at 30 minutes (headline 7.65%). The one weak spot is the last hour on expiry days: 19 of the 20
refits pass, one has p 0.052, and the headline's p of 0.010 there was the smallest of all 21 seed sets.

*Note added 3 November 2026 after study P116:* here "mcsGARCH" and "the anchor paper's mcsGARCH" mean our
implementation (a separate time-of-day curve for expiry days, a 5-minute smoothing of the curves, a zero mean), not
the paper's own specification. Against the paper's specification (one curve, no smoothing, ARMA(1,1) mean) see
`docs/P116_BENCHMARK_AS_PRINTED.md`; there the headline hybrid's Nifty gain is 4.43% and 6.64% at 15 and 30 minutes.
P116 notes that this study's seed-to-seed range (0.28 and 0.35 points) is well below the change from switching the
benchmark (1.37 and 1.00 points).

**REAL DATA.** Every number below comes from real Indian market data: third-party 1-minute Nifty bars (checked
day by day against NSE's official values), NSE's F&O bhavcopies, NSE's participant-wise open interest file and
India VIX, the same inputs as the headline run (`results/pipeline_nse_v2/`). Nothing is simulated. Every
number is taken from a file in `results/seed_bootstrap/`. Plan: `docs/prereg/P99_SEED_BOOTSTRAP.md`, committed
and pushed (64c6151, 3 November 2026, 03:22 UTC, which is 08:52 IST) before any outcome of this study was
computed. The run started a few seconds after the push and took 71 minutes.

## 1. The answer in plain words

**The question.** Our headline says that a hybrid model (the anchor paper's mcsGARCH, multiplied by a
correction learnt by LightGBM) forecasts Nifty volatility better than mcsGARCH alone. LightGBM uses random
numbers while it learns (each step looks at a random 80% of the rows and of the features). Change the random
seed and you get a slightly different model with slightly different forecasts. The headline used one set of
seeds, and its p-values treat that one fitted model as the only possible one. A referee asked: would the
result still hold with other seeds? (`docs/REFEREE_RESPONSE_FORECAST.md`, M4.3, where we had estimated the
answer but not run it.)

**What we did.** We wrote the test down first, then reran the hybrid 20 times with 20 new sets of 5 seeds
(100 seeds in all, none used before), changing nothing else, and computed for each rerun the same gain and the
same p-value as the headline.

**What we found.**

* **The main test passed (registered rule):** at 15 and 30 minutes every one of the 20 reruns beats mcsGARCH
  with p below 0.05, and in fact below 1e-7. The seeds move the gain by about a tenth of a percentage point.
* **The headline was not a lucky draw at 15 or 30 minutes.** At 15 minutes it sits in the middle of the 20
  reruns (8th best of 21). At 30 minutes it is the *least* favourable of all 21: every new seed set gives a
  slightly larger gain (median 7.93% against the headline's 7.65%).
* **A p-value that counts both kinds of chance** (which days we happened to see, and which seeds we happened to
  draw) is below 0.001 at 15 and 30 minutes: with the draws re-centred to a true gain of zero, none of 10,000
  bootstrap draws reached the gain actually observed.
* **Last hour, all days: holds** (gains 5.60% to 6.51%, largest p 0.00043).
* **Expiry days at 15 and 30 minutes: hold** (largest p 0.0013 and 0.0004).
* **Last hour on expiry days: weakened.** 19 of the 20 reruns pass, but one (seed set 102) has p 0.052, and the
  median p over the 20 is 0.026. The headline's p of 0.0097 in this cell is the smallest of all 21 seed sets,
  so it overstated how sure we can be there. The combined bootstrap p for this cell is 0.023. This is the same
  cell that the referee response already called fragile; the claim "also on expiry days" was already dropped
  for the last hour.

## 2. What was registered

From `docs/prereg/P99_SEED_BOOTSTRAP.md` (committed before any outcome):

* **Design.** Rerun the Nifty v2 walk-forward of `hybrid_base` (and the deterministic `mcsgarch`) with K = 20
  new 5-seed ensembles, base seeds 101 to 120 (LightGBM seeds base + 1000 x s, s = 0 to 4), at all three
  targets, on the same 942 days and with every other setting of the headline run. The base seed also seeds
  the tuning pass that picks the number of boosting rounds, so each rerun re-estimates the whole hybrid. The
  headline's own seed (7) is rerun too, as a reproduction check, and is not one of the 20.
* **Statistics.** For each rerun: the QLIKE margin (100 x (hybrid's mean QLIKE / mcsGARCH's - 1); negative
  means the hybrid is better) and the one-sided DM p, computed by the same functions as the headline files.
  Over the 20: minimum, median and maximum margin and the largest p. And a **seed bootstrap p**: 10,000 draws,
  each resampling the test days in blocks (stationary bootstrap, mean block 10 days) and picking one of the 20
  reruns at random; p is the share of draws (plus one, over 10,001) whose gain, re-centred to a true gain of
  zero, is at least as large as the observed average gain.
* **Primary rule.** The headline holds if, at 15 AND 30 minutes on all days, every one of the 20 reruns has a
  negative margin and p below 0.05. It fails as "weakened" if some p is 0.05 or more with every margin still
  negative, and as "sign flips" if some rerun is no better than mcsGARCH.
* **Secondary checks** (own labels, cannot change the verdict): S1, the seed bootstrap p below 0.05 at 15 and
  30 minutes; S2, the same rule in the last hour; S3, the same rule on expiry days at each target.

## 3. Results

### 3.1 The registered cells

QLIKE margin of `hybrid_base` against `mcsgarch` (negative = hybrid better), one-sided DM p, and the seed
bootstrap p. "Worst" is the largest (least negative) margin among the 20 new reruns. Source:
`results/seed_bootstrap/summary.csv`, `verdict.csv`.

| Target | Days | Headline (seed 7): margin, p | 20 reruns: best | median | worst | Largest p | p < 0.05 | Seed bootstrap p | Rule |
|---|---|---|---|---|---|---|---|---|---|
| **next 15 min, all days** | 792 | -5.80%, 1.8e-9 | -5.92% | -5.75% | -5.64% | 1.1e-8 | 20 of 20 | below 0.001 | **holds** |
| **next 30 min, all days** | 792 | -7.65%, 3.6e-15 | -8.11% | -7.93% | -7.76% | 1.8e-13 | 20 of 20 | below 0.001 | **holds** |
| last hour, all days | 792 | -6.36%, 0.00010 | -6.51% | -5.96% | -5.60% | 0.00043 | 20 of 20 | 0.0006 | holds |
| next 15 min, expiry days | 169 | -6.64%, 0.0010 | -6.96% | -6.42% | -6.18% | 0.0013 | 20 of 20 | 0.0022 | holds |
| next 30 min, expiry days | 169 | -6.23%, 0.00021 | -7.04% | -6.69% | -6.26% | 0.00040 | 20 of 20 | 0.0004 | holds |
| last hour, expiry days | 169 | -7.91%, 0.0097 | -8.20% | -7.23% | -6.37% | 0.052 | 19 of 20 | 0.023 | weakened |

"Below 0.001" means p = 1/10,001: not one of the 10,000 draws reached the observed gain. **Primary verdict:
HOLDS. S1: passed** (seed bootstrap p below 0.05 at 15 and 30 minutes). **S2: holds. S3: holds at 15 and 30
minutes, weakened in the last hour.**

The last hour on expiry days in more detail (`ensembles.csv`): the 20 reruns have p between 0.015 and 0.052,
median 0.026; none is below 0.01; 10 of the 20 are below 0.025. Their margins run from -8.20% to -6.37%. The
headline's seed set gives -7.91% with p 0.0097, the smallest p of all 21 seed sets. So for this cell a fairer
summary than "p 0.010" is "median p 0.026, between 0.015 and 0.052 depending on the seeds".

### 3.2 Descriptive subsets (no rule)

| Target | Subset | Days | Headline margin, p | 20 reruns: best | median | worst | Largest p | Seed bootstrap p |
|---|---|---|---|---|---|---|---|---|
| next 15 min | before the closing auction | 729 | -5.52%, 1.7e-20 | -5.72% | -5.49% | -5.31% | 1.1e-18 | below 0.001 |
| next 15 min | hold-out (from 1 July 2025) | 330 | -6.38%, 0.00096 | -6.67% | -6.36% | -6.23% | 0.0018 | 0.0044 |
| next 30 min | before the closing auction | 729 | -6.86%, 5.6e-16 | -7.17% | -6.98% | -6.80% | 2.8e-16 | below 0.001 |
| next 30 min | hold-out | 330 | -9.24%, 2.8e-7 | -10.25% | -9.88% | -9.59% | 7.1e-6 | below 0.001 |
| last hour | before the closing auction | 729 | -7.01%, 0.0017 | -7.20% | -6.98% | -6.52% | 0.0036 | 0.0032 |
| last hour | hold-out | 330 | -7.10%, 6.5e-5 | -7.41% | -6.46% | -5.85% | 0.00046 | 0.0009 |

All 20 reruns have p below 0.05 in every one of these cells.

### 3.3 The seed bootstrap in more detail

`bootstrap.csv` gives, for each cell, the spread of the margin over the 10,000 draws (which mixes day sampling
and seed choice) and two variants. Registered cells, mean block 10 days:

| Cell | p (registered) | Share of draws with no gain | Margin, 2.5% point | 95% point | 97.5% point | p, headline seeds only |
|---|---|---|---|---|---|---|
| next 15 min, all days | below 0.001 | below 0.001 | -7.08% | -4.68% | -4.50% | below 0.001 |
| next 30 min, all days | below 0.001 | below 0.001 | -9.45% | -6.66% | -6.44% | below 0.001 |
| last hour, all days | 0.0006 | 0.0002 | -10.09% | -3.31% | -2.82% | 0.0008 |
| next 15 min, expiry days | 0.0022 | below 0.001 | -9.52% | -3.99% | -3.64% | 0.0027 |
| next 30 min, expiry days | 0.0004 | below 0.001 | -9.99% | -3.93% | -3.47% | 0.0002 |
| last hour, expiry days | 0.023 | 0.029 | -15.43% | -0.84% | +0.23% | 0.010 |

("Below 0.001" in the second column also means 1/10,001.) The last column resamples the days in the same way
but keeps the headline's seeds fixed. In the last hour on expiry days the 95% one-sided bound on the margin is
still negative (-0.84%), but the two-sided 95% interval (-15.43% to +0.23%) just includes zero. With mean
blocks of 5 or 20 days instead of 10 the registered p-values barely move: the largest is 0.027 (last hour,
expiry days), and every other registered cell stays at or below 0.0025.

### 3.4 How large is the seed noise?

Standard deviation of the margin over the 20 reruns (`summary.csv`, column `margin_sd`), in percentage points
of mcsGARCH's QLIKE:

| | next 15 min | next 30 min | last hour |
|---|---|---|---|
| all days | 0.075 | 0.112 | 0.244 |
| expiry days | 0.188 | 0.184 | 0.488 |

The comparisons in the next two paragraphs are POST HOC (added after the results were seen; no new loss was
computed). They put the measured spread next to numbers from the referee response, collected in
`results/seed_bootstrap/m43_comparison.csv` by `experiments/seed_bootstrap.py context`.

The DM standard error of the headline margin is 0.97 points at 15 minutes, 0.96 at 30 minutes and 1.71 in the
last hour on all days, and 2.12, 1.73 and 3.35 points on expiry days. The seed spread is 8% to 15% of that in
the six registered cells (column `measured_sd_over_dm_se`): the seeds move the margin by roughly a tenth of
what the choice of days does. The bootstrap says the same in its own units: at 15 minutes on all days the 20
reruns' average daily gains have a standard deviation of 0.00021 QLIKE units against 0.00262 for the bootstrap
draws as a whole (`bootstrap.csv`, `sd_between_ensembles` and `sd_draw`).

How this compares with what the referee response assumed (M4.3): its central guess for a 5-seed average was
0.073 points (three single-seed runs at 30 minutes, divided by the square root of 5). The measured value is
about that at 15 minutes on all days (0.075) but larger at 30 minutes (0.112) and in the last hour (0.244),
probably because the number of boosting rounds is picked by a single seeded tuning pass, which averaging the 5
final fits does not smooth out. The pessimistic values behind its "p with pessimistic noise" column (the 95%
upper bound for one seed, scaled up for subsets with fewer forecasts: 0.27 to 0.60 points at 15 and 30 minutes,
1.37 to 2.96 points in the last hour) were above what we measured in every cell, so the noise-adjusted p-values
quoted there were, if anything, conservative.

The share of days on which the hybrid's daily loss is lower than mcsGARCH's barely depends on the seeds: 70.6%
to 72.7% at 15 minutes, 73.9% to 75.3% at 30 minutes and 51.9% to 53.8% in the last hour, on all days.

### 3.5 Checks

* **The headline reproduces exactly.** The rerun with the headline's seed (7) gives forecasts identical, to the
  last bit, to the committed `results/pipeline_nse_v2/forecasts_<target>.csv`: 19,800 of 19,800 at 15 minutes,
  19,008 of 19,008 at 30 minutes and 792 of 792 in the last hour, for the realized values, mcsGARCH and the
  hybrid (largest relative difference 0). Its margins and DM p-values equal the committed scoreboards and tests
  (`reproduction_check.csv`). Fitting only mcsGARCH and the hybrid, instead of all 32 models, changes nothing.
* **mcsGARCH is the same in all 21 runs** (largest relative difference 0), as it must be: it uses no random
  numbers. So every difference between reruns comes from the hybrid's seeds.
* **Tests.** `tests/test_seed_bootstrap.py` (synthetic data only): 12 tests (the 11 committed with the plan and
  one for the post hoc step), all pass.

## 4. What this changes

* The 15- and 30-minute headline does not depend on the seeds, and the referee's estimation-noise item (M4.3)
  is closed for seeds: the margins move by about 0.1 points, every rerun is significant at any usual level, and
  the combined p-value is below 0.001.
* At 30 minutes the headline's 7.6% is, if anything, an understatement: a typical seed set gives 7.9%. We keep
  quoting the headline run's numbers (they are the registered, reproducible ones) and can add "7.8% to 8.1%
  across 20 other seed sets".
* In the last hour on expiry days, the p of 0.010 should not be quoted as if it were a property of the model:
  over 20 other seed sets it ranges from 0.015 to 0.052 (median 0.026). This cell was already the fragile one
  (it fails the step-down adjustment in M4.2); this study adds that it also depends on the seeds.

## 5. Caveats

1. **Seeds are only one source of estimation noise.** The refit dates (every 21 trading days), the training
   windows, the hyperparameters, the feature list and the choice of model are all held fixed. The spread over
   20 seed sets is a lower bound on how much a refit could move the result, not the whole of it.
2. **The model was chosen after the first full-sample run** (M4.2). This study says nothing about that search;
   the SPA and step-down tests do.
3. **mcsGARCH is held fixed.** Its own estimation noise is not resampled (it has no random numbers to change).
4. **The seed bootstrap is one simple way to mix two sources of noise** (one random rerun per block resample of
   days). It is not a Giacomini and White test. It is conservative by design, because each draw carries the
   seed noise of a single rerun while the observed gain averages 20.
5. **Same data as the headline.** This says how much the headline depends on the seeds, not whether it holds on
   new data (for that, see the Bank Nifty replication P91 and the forward test).
6. **20 seed sets** pin down the typical margin well, but the extremes of a larger set could be a little wider;
   in the last hour on expiry days, more seed sets would almost certainly include a few more with p above 0.05.

## 6. Deviations from the plan

None in the design, the statistics or the rule. One post hoc addition and three small points of execution:

* **Post hoc:** `m43_comparison.csv` (Section 3.4), written by a `context` step added to the script after the
  results were seen. It only places the measured seed spread next to numbers already in
  `results/referee_forecast/seed_adjusted_dm.csv`; it computes no loss and changes no registered number.
* The run read the raw files from the main checkout (`--raw /home/user/nifty-expiry-gamma-exposure/data/raw`)
  because the working copy used for the run has no `data/raw` of its own; these are the files listed in
  `data/MANIFEST.csv`, and the market loaded with the same 942 days as the headline (checked by the script).
* `run_info.json` lists the two commands without the `--plan-commit` argument that was passed; the plan commit
  is recorded in its own field.
* The plan estimated "a little over an hour"; the run took 71 minutes (8,026 seconds of computing in 2 worker
  processes, `timing.csv`).

## 7. Files and commands

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/seed_bootstrap.py run --raw data/raw --workers 2 \
    > results/seed_bootstrap/run_stdout.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/seed_bootstrap.py analyze --plan-commit 64c6151314bd532162a54219069d08a2e4597cb3 \
    > results/seed_bootstrap/analyze_stdout.txt 2>&1
# post hoc, Section 3.4
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/seed_bootstrap.py context > results/seed_bootstrap/context_stdout.txt 2>&1
```

`results/seed_bootstrap/`:

* `verdict.csv`: the registered rule per cell and the primary verdict.
* `summary.csv`: per target and subset, the distribution over the 20 reruns, the headline beside it, and the
  seed bootstrap p.
* `ensembles.csv`: one row per rerun (and the headline seed), target and subset: margin, DM statistic and p,
  share of days the hybrid is better, the LightGBM seeds used.
* `bootstrap.csv`: the seed bootstrap for every target, subset and block length (5, 10, 20 days).
* `daily_losses_<target>.csv`: each day's average QLIKE for mcsGARCH and for each of the 21 hybrids; every other
  number can be recomputed from these.
* `m43_comparison.csv` (POST HOC): the measured seed spread beside the referee response's DM standard errors
  and assumed seed noise.
* `reproduction_check.csv`, `timing.csv`, `run_info.json`, `run_stdout.txt`, `analyze_stdout.txt`,
  `context_stdout.txt`.

The per-forecast files (about 80 MB) are not committed; `run` rewrites them to `data/processed/seed_bootstrap/`.
