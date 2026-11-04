# A5 pre-registration: is the hybrid's gain over mcsGARCH fragile to choices nobody tuned?

**Status.** Written on 4 November 2026 at about 02:00 UTC by an AI agent (Agent C of the backtest and
stress pass), BEFORE any outcome of this study was computed. At the time of writing the raw NSE files were
still downloading (logs_tmp/download_nse.log) and no walk-forward of this study had been started. The study
closes the item left BLOCKED in reports/BACKTEST_STRESS_REPORT.md (section A5) and
reports/EXECUTION_SUMMARY.md (step 4). All outcomes will be REAL DATA (NSE bhavcopy, participant OI, India
VIX and the validated third-party 1-minute Nifty bars, as in results/pipeline_nse_v2/run_info.json).

## In plain words

The headline says our hybrid model (mcsGARCH times a LightGBM correction learnt from 41 inputs) has a
QLIKE loss 7.65% below mcsGARCH at the 30-minute horizon on 792 test days (REAL DATA,
results/pipeline_nse_v2/scoreboard_next30.csv: hybrid_base 0.200910, mcsgarch 0.217551). Several settings of
that model were never tuned on the test days: the LightGBM learning rate, the number of leaves, the minimum
rows per leaf, the 21-day refit block, the 150-day first training window and the 5-seed average. Seeds were
already studied (P99, docs/SEED_BOOTSTRAP.md) and are NOT redone here. Here each of those settings is moved,
one at a time, the hybrid alone is refitted on exactly the same data, and the gain over mcsGARCH is
recomputed. Two extra runs add random noise to every continuous input to see whether the gain needs the
inputs to be exactly as measured.

## Perturbations (in this order; the list is cut from the END when time runs out)

| Order | Name in results | What changes against the committed run (everything else equal) |
| --- | --- | --- |
| a | baseline | nothing: must reproduce results/pipeline_nse_v2/forecasts_next30.csv exactly (hybrid_base and mcsgarch); if it does not, the maximum relative difference is reported, flagged, and the study continues |
| b | lr_half | LightGBM learning_rate 0.015 instead of 0.03 |
| c | lr_double | learning_rate 0.06 |
| d | leaves_7, leaves_31 | num_leaves 7 and 31 instead of 15 (two runs) |
| e | min_leaf_half, min_leaf_double | min_data_in_leaf 50 and 200 instead of 100 (two runs) |
| f | noise_10, noise_25 | Gaussian noise added to each of the 29 continuous inputs that live in the dataset, with standard deviation 10% then 25% of that input's own standard deviation over the first training window (days before day 150); one draw per row and column from seed 9,900,001, the same draw for the 10% and 25% runs (scaled) and the same whether the row is used for training or for testing; flags and integer codes (8 columns) and the 4 inputs computed inside the walk-forward from the mcsGARCH and implied fits are not perturbed (two runs) |
| g | block_42 | refit block 42 days instead of 21 (mcsGARCH refit schedule follows the block, as in the committed code: mcs_refit_every stays 21 so mcsGARCH is refitted at every block start) |
| h | min_train_250 | first training window 250 days instead of 150 (fewer test days: the comparison is on the days this run forecasts) |
| i | single_seed | n_seeds 1 (LightGBM seed 7 alone) instead of the 5-seed average |

Fixed in every run: target next30 first (headline 7.65%), then next15 (headline 5.80%) only if time allows;
base seed 7 (seed set 7 only); models mcsgarch and hybrid_base only (the walk-forward also fits the
implied-level mcsGARCH because two inputs come from it, as in P99); start 2023-01-01, end 2026-11-02,
holdout_start 2025-07-01, min_coverage 0.95, rate 0.065, tune off. The mcsGARCH forecast is deterministic
and does not depend on any LightGBM setting; it is fitted once per training-window start and reused across
runs through an in-memory cache (identical inputs give identical fits).

## Primary outcome and pass rule

For each run: mean QLIKE of hybrid_base and of mcsgarch over all forecast anchors (as in the scoreboard),
the gain in percent, 100 x (1 - QLIKE_hybrid / QLIKE_mcsgarch), and the repo's Diebold-Mariano test
(niftygex.evaluation.diebold_mariano, HAC variance, Harvey-Leybourne-Newbold correction, Student t) on
DAILY average losses, one-sided, that the hybrid has the lower loss.

**Pass rule (fixed now):** the gain is NOT fragile if, in every perturbation that was run, the hybrid's gain
over mcsGARCH stays positive with one-sided DM p below 0.05. The range of the gain across runs is reported
with the verdict. A single failing run means FRAGILE for that setting and the write-up says which one.

## Stopping rule

Hard budget: 65 minutes of wall clock for the whole task (code, runs and write-up), computing stops at
55 minutes. Runs go in the order a to i; whatever is not reached is marked NOT RUN, nothing is reordered to
favour a result. next15 runs only after all next30 runs that fit in the budget.

## Files

Code: experiments/a5_perturbation.py (reuses scripts/run_pipeline.py's loader, experiments/seed_bootstrap.py's
configuration and reproduction check, and niftygex.forecast.run_walkforward). Outputs:
results/a5_perturbation/daily_losses_<target>_<name>.csv, summary.csv, run_info.json. Test:
tests/test_a5_perturbation.py (synthetic data only). Write-up: docs/A5_PERTURBATION.md.
