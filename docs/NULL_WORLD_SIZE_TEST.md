# Null-world size test: does the walk-forward set-up invent a gain over mcsGARCH?

SYNTHETIC data only. Run by an AI agent (Claude) on 4 November 2026 with `experiments/null_world_size_test.py`. Outputs are in `results/null_world_size_test/`. This is a quick robustness check, not a registered study: it has no plan in `docs/prereg/` and no step in `scripts/reproduce_all.py`.

## Answer first

* **No sign that the machinery leaks or invents skill.** In 10 worlds where mcsGARCH is the true model, the hybrid never beats the true forecast (the "oracle"). It is worse than the oracle in all 10 worlds, by 4.3% to 19.8% (mean 10.1%), and the one-sided Diebold-Mariano (DM) test never favours it (0 of 10 below 0.05).
* **But "beats our fitted mcsGARCH" is not a clean test of "has learned something real".** The hybrid beats the walk-forward mcsGARCH in 8 of 10 null worlds (mean gain 1.8%, range -0.7% to 4.9%). The DM test calls this significant at 5% in 4 of 10 worlds; an honest test of a true null would do so in about 0 or 1 of 10. The reason is that the fitted mcsGARCH is itself well short of the truth: the oracle beats it by 5.7% to 19.5% (mean 10.7%, 10 of 10 significant). It has to estimate its parameters from 100 to 300 days, and it sometimes falls back to an EWMA daily variance (EWMA: a fixed-weight moving average of past squared returns). So a learned model can win a few percent simply by correcting the benchmark's estimation error.
* The audit's day-permutation control (reports/BACKTEST_STRESS_REPORT.md, A2) was rerun on 3 of these worlds. It gives hybrid gains of 2.7%, 3.4% and 9.5% (2 of 3 with DM p below 0.05). Shuffling the days breaks the daily volatility persistence that the EGARCH part of mcsGARCH relies on, so mcsGARCH is misspecified there by design. That control is therefore not a null for "gain over mcsGARCH" either.

## What was simulated

One-minute returns for 300 days of 375 minutes, from an mcsGARCH data-generating process with Gaussian shocks: return = square root of (daily variance h times diurnal factor s times intraday factor q) times shock.

* h: an EGARCH(1,1,1) with Gaussian errors, a model in which log variance responds to the size and sign of yesterday's standardized return. Its parameters come from a fit to the committed real Nifty open-to-close returns in `results/pipeline_nse_v2/daily_market.csv` (942 days): omega -0.0252, alpha 0.0775, gamma -0.0240, beta 0.9704, in percent units. Real data were used only to set these numbers.
* s: a fixed U-shaped curve over the day that sums to 1 (formula in `run_info.json`).
* q: a GARCH(1,1) with unit mean, carried over from one day to the next as McsGarch does. Alpha 0.0494 and beta 0.9447 come from the committed full-sample Nifty fit (`results/pipeline_nse_v2/mcsgarch_full_sample.csv`).
* There is no option-market effect. The option chains, VIX, participant file and volume come from `niftygex.simulate.simulate_market(kappa=0)` with the same seed and have nothing to do with these returns.
* The repo's own code path was run unchanged: `build_gex_panel`, then `build_dataset` (horizon 30, anchors every 45 minutes), then `run_walkforward` (min_train 100, block 21, one LightGBM seed). Models: naive_last, har, mcsgarch, lgbm_base and hybrid_base. Each world produces 200 test days and 1,600 forecasts.
* The oracle is the true expected 30-minute variance, given the true h and q at the anchor minute. No real forecaster could produce it (it is infeasible).
* World seeds 9,950,001 to 9,950,010; permutation seeds 9,960,001 to 9,960,003. Each world took about 20 s on one core; the whole run took 249 s. The run is deterministic: a first run before the oracle was added gave identical numbers.

## Results

Gains are the percentage fall in average QLIKE (a standard loss for variance forecasts) against the walk-forward mcsGARCH. Positive means better than mcsGARCH. The DM test is one-sided ("is the model better?") on daily average losses.

| Set | Model | Worlds | Mean gain % | Min % | Max % | DM p < 0.05 | Gain > 0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Null worlds | hybrid_base | 10 | 1.79 | -0.68 | 4.85 | 4 | 8 |
| Null worlds | har | 10 | 1.30 | -3.66 | 9.83 | 1 | 4 |
| Null worlds | lgbm_base | 10 | -3.23 | -10.57 | 3.82 | 0 | 3 |
| Null worlds | oracle (true model) | 10 | 10.66 | 5.73 | 19.51 | 10 | 10 |
| Null worlds | hybrid_base against the oracle | 10 | -10.10 | -19.76 | -4.34 | 0 | 0 |
| Day-permutation control | hybrid_base | 3 | 5.19 | 2.74 | 9.46 | 2 | 3 |
| Day-permutation control | har | 3 | 0.57 | -2.42 | 5.26 | 1 | 1 |
| Day-permutation control | lgbm_base | 3 | 1.83 | -7.02 | 8.24 | 1 | 2 |

Per-world numbers: `results/null_world_size_test/worlds.csv` and `permutation_control.csv`.

## Caveats

* The simulated process is close to the fitted mcsGARCH, but it is not identical. The repo fits Student-t errors, estimates the diurnal curve from the data, and switches to EWMA when its EGARCH sanity check fails; that fallback happened in at least one refit in 8 of the 10 worlds even though EGARCH is the true daily model. So the hybrid's leftover gain is best read as mismatch and estimation error in the benchmark, not leakage. These runs cannot separate those two sources.
* With only 300 days, the benchmark's estimation error is larger than on the real 2023 to 2026 sample. The size of the distortion on real data is therefore probably smaller, but this check does not measure it.
* 10 worlds is a small number: 4 of 10 has a wide uncertainty band, but it is clearly above the 0.5 rejections expected at the 5% level.
* What this means for the real results: a gain over our mcsGARCH shows that the hybrid forecasts better than that benchmark. On its own it does not show that the hybrid has learned information mcsGARCH lacks. The real-data headline gains (5.80% at 15 minutes and 7.65% at 30 minutes) are larger than every null-world hybrid gain here (largest 4.85%), though two HAR worlds reached 9.8%; they are of the same order. No real-data number is changed by this check.
