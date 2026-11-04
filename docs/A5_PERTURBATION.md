# A5: is the hybrid's gain over mcsGARCH fragile to settings nobody tuned?

Written by an AI agent (Agent C of the backtest and stress pass) on 4 November 2026. Registered plan:
docs/prereg/A5_PERTURBATION.md (written before any outcome was computed). All numbers below are REAL DATA
unless marked SYNTHETIC; each comes from a file named next to it. The numbers quoted here are NOT yet in
docs/number_registry.csv (scripts/trace_numbers.py); registering them is left to the coordinator.

## Answer first

**No, the gain is not fragile.** All 12 registered perturbations were run at next30 (REAL DATA,
results/a5_perturbation/summary.csv) and every one passes the registered rule: the hybrid's gain over
mcsGARCH stays positive with one-sided Diebold-Mariano p below 0.05. The gain ranges from 7.46% (a single
LightGBM seed instead of the 5-seed average) to 8.13% (learning rate doubled), against the committed
7.65%; the smallest DM statistic is 7.06 (min_data_in_leaf halved) and the largest one-sided p is
1.9e-12. Adding Gaussian noise worth 25% of each continuous input's own spread costs only 0.16 points of
gain (7.49% against 7.65%); 10% noise even raises it to 8.06%, which says the correction does not rely on
the inputs being measured exactly. Two runs also move the benchmark: with 42-day blocks mcsGARCH is
refitted half as often (its QLIKE 0.219984 instead of 0.217551) and the hybrid still gains 7.59%; with
a 250-day first window the test set shrinks to 692 days and the gain is 7.96%. In plain words: none of the
untuned choices is what makes the hybrid win; moving any of them leaves the gain between about 7.5% and
8.1% with overwhelming significance.

## What was done

The headline forecasting result is that the hybrid (our mcsGARCH forecast times the exponential of a
LightGBM correction learnt from 41 inputs) has a mean QLIKE loss 7.65% below mcsGARCH at the 30-minute
horizon over 792 test days, 10 August 2023 to 2 November 2026 (REAL DATA,
results/pipeline_nse_v2/scoreboard_next30.csv). Several settings of that model were fixed by hand and never
tuned on the test days. This study moves them one at a time, refits ONLY the hybrid (and the deterministic
mcsGARCH it multiplies, cached once per training-window start) on exactly the same 942-day market and
dataset as the headline, with the headline's base seed 7, and recomputes the gain and the repo's
one-sided Diebold-Mariano test on daily average losses (niftygex.evaluation.diebold_mariano: HAC variance,
Harvey-Leybourne-Newbold correction, Student t). Seeds were already studied in P99 (docs/SEED_BOOTSTRAP.md)
and are not redone. Pass rule, fixed in the plan: the gain stays positive with one-sided p below 0.05 in
every perturbation that was run.

Code: experiments/a5_perturbation.py. It reuses scripts/run_pipeline.py's loader and target definitions,
experiments/seed_bootstrap.py's configuration and reproduction check, and niftygex.forecast.run_walkforward.
LightGBM settings are changed by wrapping niftygex.forecast._lgbm_params for the duration of one run;
nothing under src/ was modified. Command (one process, runs in the registered order):

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/a5_perturbation.py --target next30 \
        --perturbation all --out results/a5_perturbation --stop-after 27

## Reproduction check (perturbation a)

The baseline rerun reproduced the committed next30 forecasts exactly: 19008 of 19008 anchors on 792 days
identical for y, mcsgarch and hybrid_base (maximum relative difference 0.0 for all three; REAL DATA,
results/a5_perturbation/summary.csv, columns repro_*). Its mean QLIKE (hybrid 0.200910, mcsGARCH 0.217551,
gain 7.649%, DM 7.94, one-sided p 3.6e-15) equals the committed scoreboard and ablation figures.

## Summary table (target next30, REAL DATA, results/a5_perturbation/summary.csv)

Gain % = 100 x (1 - QLIKE_hybrid / QLIKE_mcsGARCH) over all forecast anchors, as in the scoreboard. Days =
test days in the Diebold-Mariano test. Pass = gain positive and one-sided p below 0.05. Seconds = wall clock
of the walk-forward on a machine shared with five other agents (0 s = read back from the work file).

| Run | Setting changed | Days | QLIKE hybrid | QLIKE mcsGARCH | Gain % | DM stat | One-sided p | Pass | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a baseline | committed settings; must reproduce the committed forecasts | 792 | 0.200910 | 0.217551 | +7.65 | 7.94 | 3.6e-15 | yes | 0 |
| b lr_half | learning_rate 0.015 | 792 | 0.200256 | 0.217551 | +7.95 | 7.82 | 8.5e-15 | yes | 237 |
| c lr_double | learning_rate 0.06 | 792 | 0.199872 | 0.217551 | +8.13 | 7.38 | 1.9e-13 | yes | 64 |
| d leaves_7 | num_leaves 7 | 792 | 0.200770 | 0.217551 | +7.71 | 7.40 | 1.7e-13 | yes | 114 |
| d leaves_31 | num_leaves 31 | 792 | 0.200664 | 0.217551 | +7.76 | 7.99 | 2.4e-15 | yes | 107 |
| e min_leaf_half | min_data_in_leaf halved | 792 | 0.200039 | 0.217551 | +8.05 | 7.06 | 1.9e-12 | yes | 100 |
| e min_leaf_double | min_data_in_leaf doubled | 792 | 0.201145 | 0.217551 | +7.54 | 7.36 | 2.3e-13 | yes | 111 |
| f noise_10 | Gaussian noise, 10% of each continuous input's training sd | 792 | 0.200022 | 0.217551 | +8.06 | 7.22 | 5.9e-13 | yes | 152 |
| f noise_25 | Gaussian noise, 25% of each continuous input's training sd | 792 | 0.201260 | 0.217551 | +7.49 | 7.27 | 4.4e-13 | yes | 130 |
| g block_42 | refit block 42 days | 792 | 0.203295 | 0.219984 | +7.59 | 7.76 | 1.3e-14 | yes | 45 |
| h min_train_250 | first training window 250 days | 692 | 0.207003 | 0.224911 | +7.96 | 7.63 | 3.9e-14 | yes | 123 |
| i single_seed | one LightGBM seed instead of the 5-seed average | 792 | 0.201330 | 0.217551 | +7.46 | 7.89 | 4.9e-15 | yes | 37 |

Daily losses of every run: results/a5_perturbation/daily_losses_next30_<name>.csv (date, qlike_hybrid,
qlike_mcsgarch). Run details, raw-file hashes, versions and timings: results/a5_perturbation/run_info.json.
Stdout: results/a5_perturbation/run_stdout_next30.txt.

## Deviations from the plan

1. The first launch crashed after the baseline walk-forward had finished and its forecasts were saved: the
   reproduction check joined in-memory Timestamp dates against the committed file's date strings and matched
   nothing (KeyError). The join was fixed (both sides as YYYY-MM-DD strings), the script was relaunched and the
   baseline was read back from its work file (data/processed/a5_perturbation, git-ignored) rather than refitted;
   the check above was also run by hand on that work file with the same result. The crashed log is kept in the
   agent's scratch space, not in the repository.
2. The noise runs perturb the 29 continuous inputs that live in the dataset (listed in run_info.json,
   noise_columns). The 8 flags and integer codes (is_expiry, tdte, regime, weekday, is_monthly_expiry,
   day_before_expiry, bn_expiry_only, closing_auction) and the 4 inputs the walk-forward computes itself from
   the mcsGARCH and implied fits (log_mcs, log_s_window, log_mcs_iv, log_implied_window) are not perturbed; the
   plan said so, but it is worth repeating that "every numeric input" means 29 of 41 here.
3. The machine was shared with five other agents during the run (load average up to 16 on 4 cores), so the
   setup took 399 s instead of about 213 s and each walk-forward took longer than P99's 128 s. The time cap
   (--stop-after 27 minutes from launch, launch at 02:20:31 UTC) cut the list from the end, as registered.
4. next15 was not run (the plan allowed it only after all next30 runs).

## What is NOT RUN

Nothing from the registered next30 list: all 12 perturbations ran within the time cap (27.0 minutes
from launch, results/a5_perturbation/run_stdout_next30.txt). NOT RUN: the optional next15 repeat
(headline 5.80%), which the plan allowed only after next30 and for which no time was left; it can be run
with the same command and --target next15. Not studied here by design: seeds (P99), the EGARCH versus EWMA
daily-variance choice and the EWMA floor (a separate, not yet approved job), and any joint move of two
settings at once.

## Tests

tests/test_a5_perturbation.py (SYNTHETIC data only, 3 tests): the noise helper's scale on training and test
rows, determinism, the same draw at 10% and 25%, untouched flags and NaN cells, the LightGBM override
wrapper, and the registered parameter values. tests/test_reproduce_all.py passes with the new step
a5_perturbation_next30 in scripts/reproduce_all.py (8 tests). A synthetic smoke run of the whole script
(300 simulated days, 6 perturbations) was done in the scratch space before the real run; its numbers are
SYNTHETIC and are not reported.
