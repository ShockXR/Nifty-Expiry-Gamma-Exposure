# P99 pre-registration: does the forecasting headline survive a bootstrap over LightGBM seeds?

**Status.** Written on 3 November 2026 (phase 1). **This design is committed to git before any outcome of
this study has been computed.** This study has computed no forecast loss, no margin, no test statistic and no
bootstrap draw for any of its 20 new seed ensembles. The only real-data computation so far is the
reproduction check in Section 9: the headline run's own base seed (7) was rerun with the new code and its
forecasts were compared with the committed ones, forecast by forecast, without computing any loss. Those
forecasts and their losses are already public (`results/pipeline_nse_v2/`). The commit that adds this file is
the time stamp, and phase 2 (the run) starts only after that commit exists and has been pushed. The code that
phase 2 runs (`experiments/seed_bootstrap.py`) and its tests (`tests/test_seed_bootstrap.py`, synthetic data
only) are in the same commit. Phase 2 writes `results/seed_bootstrap/` and `docs/SEED_BOOTSTRAP.md`. All
results will be REAL DATA.

## In plain words

The headline forecasting result says that our hybrid model (the anchor paper's mcsGARCH multiplied by a
LightGBM correction) has a lower QLIKE loss than mcsGARCH. LightGBM uses random numbers when it fits (it
looks at a random 80% of the rows and of the features at each step), so a different random seed gives a
slightly different model and slightly different forecasts. The headline used one set of seeds. Its
Diebold-Mariano p-values treat that one fitted model as if it were the only possible one. A referee asked
whether the result would survive if the seeds had been different.

Here we simply refit the hybrid 20 more times, each time with a fresh set of 5 seeds that the headline never
used, on exactly the same data and with every other setting unchanged. For each of the 20 refits we compute
the same margin and the same p-value as the headline. **The headline holds if even the worst of the 20 still
beats mcsGARCH with p below 0.05, at both 15 and 30 minutes.** We also report one p-value that mixes the two
sources of chance: which days we happened to observe, and which seeds we happened to draw.

## Why this test

`docs/REFEREE_RESPONSE_FORECAST.md`, section M4.3 ("Estimation noise in the DM p-values"): the referee said the
DM p-values treat fitted models as fixed. We answered by treating seed noise as an extra error term, added to
the DM variance, measured from only three single-seed runs at 30 minutes
(`results/pipeline_nse_v2/seed_noise_next30.csv`), and we wrote: "A real bootstrap over seeds and refits would
rerun the walk-forward with new seeds ... the hybrid alone at most 2.2 hours. **We did not run it.**" This study
runs it, for the hybrid.

## Prior knowledge (what was known when this was written)

All REAL DATA, from committed files:

* **The headline** (`results/pipeline_nse_v2/`, run with base seed 7): `hybrid_base` against `mcsgarch`, QLIKE
  change (anchor means, `scoreboard_*.csv`) and one-sided DM p (daily means, `tests_*.csv`) on 792 test days
  (169 expiry days):

  | Target | All days | DM p | Expiry days | DM p |
  |---|---|---|---|---|
  | next 15 minutes | -5.80% | 1.8e-9 | -6.64% | 0.0010 |
  | next 30 minutes | -7.65% | 3.6e-15 | -6.24% | 0.00021 |
  | last hour | -6.36% | 0.00010 | -7.91% | 0.0097 |

* **Seed noise measured so far** (`seed_noise_next30.csv`, `results/referee_forecast/seed_noise_summary.csv`):
  three SINGLE-seed hybrids (seeds 7, 8, 9) at 30 minutes had margins -7.46%, -7.83% and -7.63% (SD 0.19
  points). Nothing has been measured for 5-seed ensembles, at 15 minutes, in the last hour or on expiry days.
* **Break-even noise** (`results/referee_forecast/seed_adjusted_dm.csv`): to push the all-days p above 5%, the
  seed noise would have to be at least 3.29 points; the smallest break-even for 1% is 0.34 points, in the last
  hour on expiry days.
* **Refit spread** across six hybrid variants that differ only by uninformative feature blocks
  (`refit_envelope.csv`): at most 0.88 points (SD), largest DM p 0.037 (last hour, expiry days).
* **Cost** (`seed_rerun_cost.csv`): at most about 2.2 hours on one core for 20 hybrid configurations.

From this, we expect the headline to pass at 15 and 30 minutes on all days: the seed noise seen so far is more
than ten times smaller than what would be needed to lose significance. The cells most at risk are on expiry
days and in the last hour (one forecast a day, so less averaging), above all the last hour on expiry days. This
study is worth running because it replaces an assumption (seed noise is an independent extra error of a
size measured from three runs at one horizon) with a direct measurement at every horizon.

## 1. Design

**What is rerun.** The Nifty v2 walk-forward of `hybrid_base` exactly as in the headline run
(`scripts/run_pipeline.py --source nse --end 2026-11-02 --n-seeds 5 --seed-study 3 --out
results/pipeline_nse_v2`; settings in `results/pipeline_nse_v2/run_info.json`), at all three targets (next 15
minutes, next 30 minutes, last hour), on the same 942 days (2 January 2023 to 2 November 2026; 792 test days
from 10 August 2023) with the same 21-day walk-forward (150 training days, every model refitted before each
block of 21 days). Only two models are fitted: `mcsgarch` (deterministic) and `hybrid_base`.
`run_walkforward` also fits the implied-level mcsGARCH (`mcsgarch_iv`) inside each block, because two of the
hybrid's features come from it (`log_mcs_iv`, `log_implied_window`); it is not scored. The market, the daily
GEX and implied panels and the datasets are built by the pipeline's own functions (`load_market` from
`scripts/run_pipeline.py`, `build_gex_panel`, `build_implied_panel`, `build_dataset`), and the script refuses to
run if any setting differs from the headline run's `run_info.json` or if the number of days or the last day
differs.

**What changes: the seeds.** In the pipeline (`src/niftygex/forecast.py`), the base seed `cfg.seed` does two
things for each LightGBM model in each block: it seeds the tuning pass that picks the number of boosting rounds
(early stopping on the last 20% of the training window), and it gives the 5 final fits their seeds
`cfg.seed + 1000 * s`, s = 0 to 4, whose log forecasts are averaged. So a new base seed refits the whole hybrid
with new randomness: the tuning pass and all 5 fits. (This adapts the suggested design: we keep the pipeline's
behaviour, so each ensemble's number of rounds can differ too. Seeds are the only thing that changes.)

* **The 20 new ensembles (K = 20):** base seeds 101, 102, ..., 120, so LightGBM seeds
  {101, 1101, 2101, 3101, 4101}, ..., {120, 1120, 2120, 3120, 4120}: 100 distinct seeds. None is used by the
  headline ({7, 1007, 2007, 3007, 4007}) or by its seed study (single seeds 7, 8 and 9), and no two ensembles
  share a seed (checked in code and in a test).
* **The headline seed (7) is rerun as well**, as a reproduction check (Section 9). It is reported next to the
  20 new ensembles but is **not one of the K = 20** in any statistic or rule below.

**Subsets.** All 792 test days and the 169 expiry days (registered). Also, descriptively and with no rule
attached: the 729 days before NSE's closing auction began (3 August 2026) and the hold-out (from 1 July 2025).
Subset masks come from `niftygex.evaluation.subgroups`, as in the pipeline.

## 2. The exact commands

From the repository root, at the commit that adds this file, with the raw data listed in `data/MANIFEST.csv`
in `data/raw` (or passed with `--raw`):

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/seed_bootstrap.py run --raw data/raw --workers 2 \
    > results/seed_bootstrap/run_stdout.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/seed_bootstrap.py analyze --plan-commit <this commit> \
    > results/seed_bootstrap/analyze_stdout.txt 2>&1
```

`run` computes 63 walk-forwards (21 base seeds times 3 targets) in 2 worker processes and stores each one's
forecasts in `data/processed/seed_bootstrap/` (git-ignored; a finished one is not recomputed if the run has to
be restarted). `analyze` computes everything below and writes `results/seed_bootstrap/`. Expected run time: see
Section 9.

## 3. Statistics reported

For each ensemble (the 20 new ones and the headline seed), target and subset, exactly as in the headline files:

* **Margin**: 100 x (mean QLIKE of `hybrid_base` / mean QLIKE of `mcsgarch` - 1), QLIKE = y/f - log(y/f) - 1
  averaged over all forecasts in the subset (as in `scoreboard_*.csv`). Negative means the hybrid is better.
* **DM test**: `niftygex.evaluation.diebold_mariano(L_mcsgarch, L_hybrid, alternative="greater")` on daily
  average QLIKE (`daily_losses`), the same function and arguments as in `tests_*.csv`: Newey-West variance,
  Harvey-Leybourne-Newbold correction, Student t with n - 1 degrees of freedom, one-sided (the hybrid has the
  lower loss). Also the DM statistic and the share of days on which the hybrid's daily loss is lower.

Over the 20 new ensembles, per target and subset: **minimum, median and maximum margin** (the maximum is the
worst ensemble), mean and standard deviation of the margin, **largest DM p**, the median p, the number of
ensembles with p < 0.05, and where the headline seed's margin falls among them.

**Seed bootstrap p** (per target and subset; registered for all days and expiry days). Let L0(t) be mcsGARCH's
daily average QLIKE on day t, Lk(t) that of ensemble k (k = 1 to 20) and dk(t) = L0(t) - Lk(t) (positive means
the hybrid is better). The observed gain is the grand mean D = average of dk(t) over all days and all 20
ensembles. Each of B = 10,000 bootstrap draws:

1. resamples the days with the stationary bootstrap of Politis and Romano (1994), mean block length 10 days
   (the same function and block length as the SPA test in `docs/REFEREE_RESPONSE_FORECAST.md`, M4.2:
   `stationary_bootstrap_indices` in `experiments/referee_forecast.py`), which keeps runs of neighbouring days
   together because losses are correlated from day to day;
2. picks ONE of the 20 ensembles at random (uniformly);
3. computes that ensemble's mean gain D* over the resampled days.

So D* varies for two reasons: which days were observed and which seeds were drawn. The **registered seed
bootstrap p** for "the hybrid is not better than mcsGARCH" is the null-imposed one-sided p,
p = (1 + number of draws with D* - D >= D) / (B + 1). It asks: if the true gain were zero, how often would one
5-seed ensemble, on resampled days, show a gain at least as large as the average gain observed? Because each
draw uses one ensemble while D averages twenty, this is conservative (it charges the full seed noise of a
single ensemble). Also reported: the percentile version (the share of draws with no gain, D* <= 0), the 2.5%,
50%, 95% and 97.5% points of the draws' margin, 100 x (mean Lk / mean L0 - 1) on the resampled days, the
standard deviation of D* and of the 20 ensembles' gains, and, with the same day draws, the p of the headline
ensemble alone (days resampled, seeds fixed), which shows how much the seed draw adds. Sensitivity (no rule):
mean blocks of 5 and 20 days. Random numbers: `numpy.random.default_rng([99, target index, subset index,
block index])`.

Also reported: the reproduction check (Section 9) and whether mcsGARCH's forecasts are identical in all 21
runs (they must be: it uses no random numbers).

## 4. Decision rule

**Primary (fixed before any outcome): the headline holds if, at next 15 minutes AND at next 30 minutes, on all
792 test days, every one of the 20 new ensembles has a negative margin (the hybrid's mean QLIKE is lower than
mcsGARCH's) and a one-sided DM p below 0.05.** Equivalently: the worst ensemble still beats mcsGARCH with
p < 0.05 at both horizons.

**What counts as the headline failing.** Any of the 40 ensemble-horizon cells (20 ensembles times 2 horizons,
all days) with a margin of zero or more, or with p of 0.05 or more. Two cases get a name:

* **"weakened"**: every ensemble still has the lower mean loss at both horizons, but at least one has p >= 0.05.
* **"sign flips"**: at least one ensemble has a mean loss at least as high as mcsGARCH's at 15 or 30 minutes.

In either case `docs/SEED_BOOTSTRAP.md` says in its first paragraph that the headline, as stated (a single
fitted hybrid significant at the 1% level or better), does not survive the seed bootstrap, and gives the
fraction of ensembles that pass.

**Secondary checks (registered, each with its own label; they cannot change the primary verdict).**

* **S1, seed bootstrap.** The seed bootstrap p (mean block 10) is below 0.05 at next 15 and at next 30 minutes,
  all days. If it is not, we say so next to the primary verdict.
* **S2, last hour.** The same rule as the primary for the last hour on all days.
* **S3, expiry days.** The same rule on the 169 expiry days, separately at each of the three targets. The last
  hour on expiry days (169 single windows; DM p 0.0097 in the headline) is the cell we expect to be most
  fragile.

No multiple-testing adjustment is applied across ensembles: the rule asks every ensemble to pass on its own,
which is stricter than any adjustment. The pre-auction and hold-out subsets, the MSE loss and the sensitivity
blocks are descriptive only.

## 5. What will be reported, whatever the result

In `docs/SEED_BOOTSTRAP.md`, for a beginner reader, every number labelled REAL DATA and taken from a file in
`results/seed_bootstrap/`:

1. The primary verdict in the first paragraph, with the minimum, median and maximum margin and the largest DM p
   at each target, and the seed bootstrap p.
2. S1 to S3 with their labels.
3. Tables per target and subset (registered and descriptive subsets), and the headline seed beside the 20.
4. The reproduction check and the mcsGARCH identity check.
5. Run time, and every deviation from this plan.

Files: `ensembles.csv` (one row per ensemble, target and subset), `summary.csv`, `bootstrap.csv`, `verdict.csv`,
`reproduction_check.csv`, `daily_losses_<target>.csv` (each ensemble's daily average QLIKE, from which every
other number can be recomputed), `timing.csv`, `run_info.json` and the two stdout logs.

## 6. Deviations policy

* After this file is committed, the design does not change. Any change is a deviation, listed in
  `docs/SEED_BOOTSTRAP.md` with the reason and whether any outcome had been seen when it was decided.
* A crash is fixed with the smallest change that makes the run finish, and only if it changes no number of the
  jobs that already finished.
* If the reproduction check fails in phase 2 (the rerun of seed 7 no longer matches the committed forecasts),
  that is reported, the cause is looked for, and the verdict is still given on the 20 new ensembles, with the
  failure stated next to it.
* If the run cannot be completed, that is reported with the reason, and no verdict is given.

## 7. Known weaknesses of this test

1. **Seeds are only one source of estimation noise.** The refit dates (every 21 trading days), the training
   windows, the hyperparameters (fixed in the development window), the feature list and the model choice are
   all held fixed. The variation across 20 seed sets is therefore a lower bound on the hybrid's estimation
   uncertainty, not all of it. A bootstrap over refit schedules or hyperparameters would need another design.
2. **The model was chosen after the first full-sample run** (`docs/REFEREE_RESPONSE_FORECAST.md`, M4.2). No
   test on these data can undo that; the SPA and step-down results address the search over models, this study
   addresses the seeds only.
3. **mcsGARCH is held fixed.** Its estimation noise (and the EWMA fallback in two refits, M5) is not
   resampled; it is the same deterministic benchmark in every run.
4. **The bootstrap p mixes two sources of noise in one simple way** (one random ensemble per day draw). It is
   not a Giacomini and White test, and its validity rests on the stationary bootstrap for the day dimension and
   on the 20 ensembles being a fair sample of possible seeds.
5. **Same data as the headline.** This study says nothing about new data; it says how much of the headline
   depends on the particular random seeds.

## 8. Code

* `experiments/seed_bootstrap.py`, new. `run` loads the market with the pipeline's `load_market`, builds the
  panels and datasets once, then forks 2 worker processes (`maxtasksperchild=1`), each calling
  `niftygex.forecast.run_walkforward` with the headline's `WalkForwardConfig` except `models = ["mcsgarch",
  "hybrid_base"]` and `seed` = the base seed. `check` compares the base-seed-7 forecasts with the committed ones
  and computes no loss. `analyze` does Sections 3 and 4. No file of the pipeline or of the package is changed,
  and no default of `scripts/run_pipeline.py` changes.
* `tests/test_seed_bootstrap.py`, new, synthetic data only: the seed sets do not overlap; the configuration
  matches the headline's `run_info.json`; fitting only mcsGARCH and the hybrid gives exactly the same hybrid
  and mcsGARCH forecasts as a run with ten models and the same seed, mcsGARCH is identical under a different
  seed and the hybrid is not; the work files round-trip exactly and the reproduction check detects a changed
  forecast; the per-ensemble margin and DM p equal the pipeline's `scoreboard` and `diebold_mariano`; the
  daily-loss file rebuilds the anchor-level margin; the seed bootstrap equals a plain day bootstrap when
  K = 1 or when all ensembles are identical, gives p near 0 for a clear gain and near 1 for a clear loss,
  widens and raises p when ensembles disagree, and rejects about 5% of the time under the null (less with
  seed noise); and the decision rule's three labels.

## 9. Checks run before this file was committed (3 November 2026)

* **The headline seed reproduces exactly.** With the code in this commit, the headline's own base seed was
  rerun at all three targets, writing to a scratch folder:

  ```
  PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/seed_bootstrap.py run --raw <main checkout>/data/raw \
      --seeds 7 --workers 2 --work <scratch folder>
  PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/seed_bootstrap.py check --work <scratch folder>
  ```

  The market loaded with 942 days (2 January 2023 to 2 November 2026) and every setting matched the headline
  run's `run_info.json`. Compared with the committed `results/pipeline_nse_v2/forecasts_<target>.csv`, read
  back at full precision, **every forecast is identical to the last bit**: 19,800 of 19,800 at next 15 minutes,
  19,008 of 19,008 at next 30 minutes and 792 of 792 in the last hour, for the realized values `y`, for
  `mcsgarch` and for `hybrid_base` (largest relative difference 0 in every case). So fitting only mcsGARCH and
  the hybrid, instead of all 32 models, does not change the hybrid, and the 20 new ensembles differ from the
  headline only through their seeds. `check` computes no loss, and no loss was computed.
* **Run time.** The three walk-forwards for seed 7 took 147 s (next 15), 129 s (next 30) and 96 s (last hour)
  in 2 worker processes; loading the market and building the panels and datasets took 172 s. The full run
  (21 base seeds times 3 targets) should therefore take about 21 x 372 s / 2, a little over an hour, plus 3
  minutes of preparation.
* **Tests.** `PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_seed_bootstrap.py`: 11 passed. In
  the size test (200 simulated data sets of 150 days with an AR(1) loss differential whose true mean is zero,
  499 draws each, mean block 5) the 5% test rejected 3.5% of the time with identical ensembles and 1.5% of the
  time with seed noise, so the bootstrap p is not too liberal and is conservative when ensembles differ, as
  intended.
