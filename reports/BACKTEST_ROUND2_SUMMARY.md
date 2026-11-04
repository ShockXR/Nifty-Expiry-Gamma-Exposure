# Backtest round 2: consolidated summary

Date: 4 November 2026. Everything below was run and written by AI agents (Claude), not by a human. Each number is quoted from a committed file of this branch. No printed headline number was changed this round. REAL DATA means NSE files and the committed real-data forecasts; SYNTHETIC means simulated data.

## 1. What this round did

1. We stress-tested the headline forecast result again on the committed forecast files: by market regime, under other loss functions, against data snooping across the 32-model ladder, and on Bank Nifty and Fin Nifty. A second agent then reviewed those statistics independently.
2. After merging pull requests #1 to #3, we checked the repository's health (merge, tests, number tracer, documents, CI) and compared the freshly downloaded raw NSE files with the data manifest, hash by hash.
3. We reran 31 quick reproduction steps in a scratch copy. Steps that read committed outputs reproduce; steps that read the raw files do not, because the raw folder on disk is not the committed snapshot.
4. We reviewed and carried the GEX input checks of PR #4, ran a live demonstration on 4 November 2026 (the forecast freeze was refused by design, since it was not an expiry day), and gave the P112 forward test a guarded runner.
5. We built a cleaner null test (worlds where mcsGARCH is the true model). It shows the machinery does not invent skill, but that a significant gain over a fitted mcsGARCH overstates how significant the evidence is.

## 2. Workstreams

| Workstream | Question | Answer in one line | Status | Read |
| --- | --- | --- | --- | --- |
| Extended backtests, Nifty (REAL DATA) | Does the gain survive slices, other losses and data-snooping tests? | Yes: positive in all 34 regime cells at 15 and 30 minutes, SPA p 0.000 / 0.000 / 0.005; not significant under variance-scale MSE | VERIFIED (calibration PARTIALLY VERIFIED, MSE FAILED) | docs/BACKTEST_EXTENDED.md |
| Extended backtests, Bank Nifty and Fin Nifty (REAL DATA) | Does it replicate on the other indices? | Yes at 15 and 30 minutes on both; last hour holds on Bank Nifty (8.1%, p = 3e-7) but not on Fin Nifty (1.0%, p = 0.30) | VERIFIED, last hour on Fin Nifty FAILED | docs/BACKTEST_EXTENDED.md |
| Independent review of the backtest statistics | Are the methods right, and do the tests have the right size? | Methods match their textbook definitions; size near 5% (SYNTHETIC); spot numbers reproduce | VERIFIED | reports/BACKTEST_EXTENDED_REVIEW.md |
| Reproduction rerun (31 steps) | Do the quick steps reproduce the committed files? | Committed-output steps reproduce (largest difference 6.8e-12); raw-data steps differ because of the data on disk; two crashed on the new data and were fixed (holiday-shifted expiries, an --end date) | VERIFIED for committed-output steps, FAILED for raw-data steps (two fixed) | reports/REPRODUCTION_RERUN.md |
| Post-merge health | Is main complete and healthy? | All three PRs are on main; number tracer 664 MATCH, 0 MISMATCH; CI green on 3.11 to 3.13 | VERIFIED (tests PARTIALLY VERIFIED: one failure from uncommitted A5 work at the time) | reports/POST_MERGE_HEALTH.md |
| Raw-data integrity | Do the downloaded files match the manifest? | 2,833 of 5,213 match; the 4 that differ match exactly once the 3 November rows are dropped; 2,376 not downloaded | VERIFIED for downloaded files, NOT RUN for the rest | reports/POST_MERGE_HEALTH.md |
| Live demo, 4 November | Can we freeze a forecast before the open? | No: the script refuses non-expiry days; the 3 November re-score is identical | PARTIALLY VERIFIED | reports/LIVE_DEMO_2026-11-04.md |
| PR #4 GEX input checks | Do they reject valid real chains or change numbers? | No false rejection on 20 real chains, identical numbers, about 9 s extra per panel; four low-severity defects | VERIFIED | reports/PR4_GEX_REVIEW.md |
| GEX validation fixes on top of PR #4 (SYNTHETIC cases, REAL panels) | Which broken inputs still go through without a warning? | 41 stress cases: before any fix 3 detected, 11 safely handled, 24 silently ignored, 3 crashed; with PR #4 23 / 14 / 4 / 0; with the four fixes on top 25 / 14 / 2 / 0 (the two left are unit mistakes the engine cannot see). Nifty and Bank Nifty GEX panels rebuilt from the raw files are byte-identical under the old engine, PR #4 and PR #4 plus fixes | VERIFIED (units FAILED by design, stays under R-A5) | reports/GEX_VALIDATION_FIX.md |
| Null-world size test (SYNTHETIC) | Does the walk-forward invent a gain over mcsGARCH? | It never beats the true model (0 of 10), but beats the fitted mcsGARCH significantly in 4 of 10 null worlds | VERIFIED | docs/NULL_WORLD_SIZE_TEST.md |
| A5 perturbation (REAL DATA, pre-registered) | Is the 30-minute gain fragile to settings nobody tuned? | No: the baseline reproduces the committed forecasts exactly (7.65%), and all 12 perturbations (learning rate, leaves, leaf size, 10% and 25% input noise, block 42, min_train 250, one seed) keep the gain between 7.46% and 8.13%, every one-sided p at most 1.9e-12 | VERIFIED (15-minute target NOT RUN) | docs/A5_PERTURBATION.md |
| P120 joint option-market test (REAL DATA, pre-registered, run under its stopping rule) | Is the gain free of option-market inputs when both groups are removed together? | A joint contribution of 1 point or more is ruled out only on Nifty at 30 minutes (+0.06 points, 90% interval -0.56 to +0.68) and Bank Nifty at 15 minutes (-0.18, -0.67 to +0.32); Nifty 15 minutes (+0.36, -0.37 to +1.09) and Bank Nifty 30 minutes (-0.20, -0.90 to +0.49) are not shown either way; no row is worse by 1 point. Surprise: at the last hour on Bank Nifty the hybrid is 2.4 points BETTER without the option-market inputs (all five seed sets, two-sided p 0.02) | PARTIALLY VERIFIED (Fin Nifty rows, pooled rows and the Nifty last hour NOT COMPUTED) | docs/P120_JOINT_OPTION_MARKET.md |

## 3. What changed in the code

* **GEX engine input checks (PR #4, carried on this branch):** `validate_chain` refuses missing columns, invalid option types, negative OI or IV, and non-positive strike, spot, lot size, time or forward, instead of silently returning a net GEX of 0. Four small gaps were closed on top of it, each marked in `src/niftygex/gex.py`: the near-the-money gross total now gives NaN (not 0.0) on an all-missing chain, infinite open interest and an IV above the solver's own limit (500% a year) are refused, and a chain with only calls, only puts or no strike within 2% of spot gets a warning. No committed GEX panel changed (rebuilt from the raw files for Nifty and Bank Nifty: byte-identical).
* **P112 forward-test runner:** `scripts/forward_test_p112.py` counts the forward days collected and refuses to compute any outcome before the registered stopping point.
* **CI:** a fourth job, `pytest-pinned` (Python 3.13), installs `requirements-tested.txt`, so the versions that produced the committed numbers are tested too. The three unpinned jobs are unchanged.
* **README:** the Quick start installs the `download` extra (it holds `requests`) before the real-data commands.
* **New experiment scripts:** `experiments/backtest_extended.py` and `experiments/null_world_size_test.py`; `experiments/a5_perturbation.py` (registered in `scripts/reproduce_all.py`). No change to the forecasting code itself.
* **Expiry event study:** `experiments/expiry_event_study.py` no longer reads a holiday-shifted monthly expiry as a schedule change (the crash found by the reproduction rerun), and it and `experiments/referee_expiry.py` take an `--end` date (default 2026-11-02) so the committed results reproduce from a refreshed download. On the committed data the detector gives the same seven changes and the same `schedule_changes_detected.csv` and `events.csv`.

## 4. Four findings for the professor meeting

1. **Beating a fitted mcsGARCH overstates significance.** In 10 SYNTHETIC worlds where mcsGARCH is the true model, the hybrid is worse than the true forecast in all 10 (by 4.3% to 19.8%), so there is no leakage. But it beats our fitted mcsGARCH by 1.8% on average, and the Diebold-Mariano test calls that significant in 4 of 10 worlds where 0 or 1 is expected. The fitted benchmark is about 10.7% short of the truth, because of estimation error and its EWMA fallback. A gain over mcsGARCH shows the hybrid forecasts better than that benchmark. It does not, on its own, show that the hybrid learned information mcsGARCH lacks. The real gains (5.80% and 7.65%) are larger than every null-world hybrid gain (largest 4.85%), but of the same order.
2. **The extended backtests hold on all three indices at 15 and 30 minutes; the last hour fails on Fin Nifty.** On Nifty the gain is positive in every regime cell and every rolling 63-day window at 15 and 30 minutes, and passes Hansen's SPA test. Bank Nifty and Fin Nifty replicate at 15 and 30 minutes. At the last hour the gain is 8.1% (p = 3e-7) on Bank Nifty but 1.0% (p = 0.30) on Fin Nifty, where the hybrid loses on some expiry days. Under variance-scale MSE the gain is not significant, because a few days in June 2024 dominate that loss.
3. **Raw-data steps read every file on disk.** The bhavcopy-based studies had no fixed end date: they used whatever was in `data/raw`. The rerun panels held five more days than the committed ones (to 3 November instead of 27 October), and the `extra_2022/` folder was missing, so those studies could not be reproduced to the committed numbers, and two of them crashed (a holiday-shifted Bank Nifty expiry read as a schedule change). The expiry event study is fixed and takes an end date now; the other raw-data studies still need the manifest snapshot (or the same end-date option) to reproduce.
4. **The option-market inputs are not the source of the gain, but the equivalence test only half settles it.** P120 rules out a joint contribution of 1 point or more on Nifty at 30 minutes and Bank Nifty at 15 minutes; the other two primary rows are not shown either way, Fin Nifty was not run. The unqualified sentence "the gain does not come from option-market information" may not be used; "the data rule out a contribution of 1 point or more on two of the four rows computed" may. On Bank Nifty's last hour the hybrid is better without those inputs, which is one more reason to keep the last-hour result out of the headline.

## 5. Open items and what was deliberately not done

* **Not done on purpose:** the rerun of the headline pipeline with a causal EGARCH choice and EWMA floor (risk R-A1) was not approved, so it was not run. No printed headline number changed.
* P120: run Fin Nifty (needs `scripts/fetch_finnifty_data.py`, which in turn needs the `nse_full/` download), the pooled rows and the Nifty last hour; then the PENDING entry in `scripts/reproduce_all.py` can go.
* A5: the 15-minute target (same command with `--target next15`).
* Download `extra_2022/` and the Fin Nifty, full-bhavcopy and Sensex raw files, then rerun the raw-data reproduction steps against the manifest snapshot; give the other raw-data studies an `--end` option like the expiry event study.
* The live freeze needs the daily pipeline files extended past 2 November before the next expiry (9 November); the P112 runner still lacks the code that carries the pinned walk-forward forward.
* `features.build_gex_panel` sums its near-the-money columns with pandas itself, so those columns would still read 0.0 on an all-NaN subset; `gex_by_strike` crashes on string option types (both outside the engine; low severity).
* PR #4's four low-severity points (reports/PR4_GEX_REVIEW.md, section 4).
* Watch the first run of the pinned CI job; bump `actions/checkout` and `actions/setup-python` to Node 24 versions.
* New numbers in this round are not yet in `docs/number_registry.csv`; add adapters before quoting them in the pitch.

## 6. How to rerun the new scripts

```bash
# Extended backtests (REAL DATA, committed forecasts only; seconds)
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/backtest_extended.py \
    --run results/pipeline_nse_v2 --p116 results/p116_benchmark_as_printed \
    --out results/backtest_extended --draws 5000 --block 10 --seed 9700001
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/backtest_extended.py --index banknifty
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/backtest_extended.py --index finnifty

# Null-world size test (SYNTHETIC; about 4 to 8 minutes on one core)
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/null_world_size_test.py 10 300 3

# A5 perturbation (REAL DATA, needs data/raw; long)
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/a5_perturbation.py --target next30 \
    --perturbation all --out results/a5_perturbation --stop-after 27

# P112 forward test: count collected days; run refuses before the stopping point
PYTHONPATH=src python scripts/forward_test_p112.py count --raw data/raw

# P120 (REAL DATA, needs data/raw; about 30 minutes for Bank Nifty with 2 workers)
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py run --index BANKNIFTY --workers 2
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p120_joint_option_market.py analyze --plan-commit 30da69e

# Raw-data integrity
python scripts/data_manifest.py check
```

These commands write into `results/`, so run them in a copy if you need to keep the committed files untouched. `scripts/reproduce_all.py` overwrites committed results as well.
