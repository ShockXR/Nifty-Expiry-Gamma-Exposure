# Backtest round 2: consolidated summary (draft)

Date: 4 November 2026. Everything below was run and written by AI agents (Claude), not by a human. Each number is quoted from a committed file of this branch, or, where marked "[in progress, numbers to be confirmed]", from a file another agent was still writing. No printed headline number was changed this round. REAL DATA means NSE files and the committed real-data forecasts; SYNTHETIC means simulated data.

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
| Reproduction rerun (31 steps) | Do the quick steps reproduce the committed files? | Committed-output steps reproduce (largest difference 6.8e-12); raw-data steps differ because of the data on disk; two crash on the new data | VERIFIED for committed-output steps, FAILED for raw-data steps | reports/REPRODUCTION_RERUN.md |
| Post-merge health | Is main complete and healthy? | All three PRs are on main; number tracer 664 MATCH, 0 MISMATCH; CI green on 3.11 to 3.13 | VERIFIED (tests PARTIALLY VERIFIED: one failure from uncommitted A5 work at the time) | reports/POST_MERGE_HEALTH.md |
| Raw-data integrity | Do the downloaded files match the manifest? | 2,833 of 5,213 match; the 4 that differ match exactly once the 3 November rows are dropped; 2,376 not downloaded | VERIFIED for downloaded files, NOT RUN for the rest | reports/POST_MERGE_HEALTH.md |
| Live demo, 4 November | Can we freeze a forecast before the open? | No: the script refuses non-expiry days; the 3 November re-score is identical | PARTIALLY VERIFIED | reports/LIVE_DEMO_2026-11-04.md |
| PR #4 GEX input checks | Do they reject valid real chains or change numbers? | No false rejection on 20 real chains, identical numbers, about 9 s extra per panel; four low-severity defects | VERIFIED | reports/PR4_GEX_REVIEW.md |
| GEX validation fixes on top of PR #4 [in progress, numbers to be confirmed] | Which broken inputs still go through without a warning? | 41 SYNTHETIC stress cases: 23 detected, 14 safely handled, 4 silently ignored before the fixes; real GEX panels byte-identical | PARTIALLY VERIFIED (full suite status not yet filled in) | reports/GEX_VALIDATION_FIX.md |
| Null-world size test (SYNTHETIC) | Does the walk-forward invent a gain over mcsGARCH? | It never beats the true model (0 of 10), but beats the fitted mcsGARCH significantly in 4 of 10 null worlds | VERIFIED | docs/NULL_WORLD_SIZE_TEST.md |
| A5 perturbation (REAL DATA) [in progress, numbers to be confirmed] | Is the 30-minute gain fragile to untuned settings? | So far the baseline reproduces (7.65%); learning rate halved 7.95%, doubled 8.13%, 7 leaves 7.71% | NOT RUN to the end (4 variants on disk) | docs/prereg/A5_PERTURBATION.md, results/a5_perturbation/summary.csv |
| P120 joint option-market test [in progress] | Is the gain free of option-market inputs? | Bank Nifty runs were still going (12 of 15 jobs done at last log); no outcome on disk | NOT RUN | results/p120_joint_option_market/ |

## 3. What changed in the code

* **GEX engine input checks (PR #4, carried on this branch):** `validate_chain` refuses missing columns, invalid option types, negative OI or IV, and non-positive strike, spot, lot size, time or forward, instead of silently returning a net GEX of 0. Agent B's additions on top (IV cap, infinite OI, coverage warnings) are marked in `src/niftygex/gex.py` [in progress, numbers to be confirmed]. No committed GEX panel changed.
* **P112 forward-test runner:** `scripts/forward_test_p112.py` counts the forward days collected and refuses to compute any outcome before the registered stopping point.
* **CI:** a fourth job, `pytest-pinned` (Python 3.13), installs `requirements-tested.txt`, so the versions that produced the committed numbers are tested too. The three unpinned jobs are unchanged.
* **README:** the Quick start installs the `download` extra (it holds `requests`) before the real-data commands.
* **New experiment scripts:** `experiments/backtest_extended.py` and `experiments/null_world_size_test.py`; `experiments/a5_perturbation.py` [in progress]. No change to the forecasting code itself.

## 4. Three findings for the professor meeting

1. **Beating a fitted mcsGARCH overstates significance.** In 10 SYNTHETIC worlds where mcsGARCH is the true model, the hybrid is worse than the true forecast in all 10 (by 4.3% to 19.8%), so there is no leakage. But it beats our fitted mcsGARCH by 1.8% on average, and the Diebold-Mariano test calls that significant in 4 of 10 worlds where 0 or 1 is expected. The fitted benchmark is about 10.7% short of the truth, because of estimation error and its EWMA fallback. A gain over mcsGARCH shows the hybrid forecasts better than that benchmark. It does not, on its own, show that the hybrid learned information mcsGARCH lacks. The real gains (5.80% and 7.65%) are larger than every null-world hybrid gain (largest 4.85%), but of the same order.
2. **The extended backtests hold on all three indices at 15 and 30 minutes; the last hour fails on Fin Nifty.** On Nifty the gain is positive in every regime cell and every rolling 63-day window at 15 and 30 minutes, and passes Hansen's SPA test. Bank Nifty and Fin Nifty replicate at 15 and 30 minutes. At the last hour the gain is 8.1% (p = 3e-7) on Bank Nifty but 1.0% (p = 0.30) on Fin Nifty, where the hybrid loses on some expiry days. Under variance-scale MSE the gain is not significant, because a few days in June 2024 dominate that loss.
3. **Raw-data steps read every file on disk.** The bhavcopy-based studies have no fixed end date: they use whatever is in `data/raw`. The rerun panels held five more days than the committed ones (to 3 November instead of 27 October), and the `extra_2022/` folder was missing, so those studies could not be reproduced to the committed numbers. Nothing points to their code. To reproduce them, restore the manifest snapshot (or add an end-date option to those steps).

## 5. Open items and what was deliberately not done

* **Not done on purpose:** the rerun of the headline pipeline with a causal EGARCH choice and EWMA floor was not approved, so it was not run. No printed headline number changed.
* Finish A5 and P120 and replace the bracketed numbers above with the final ones.
* Fill in the full-suite status in reports/GEX_VALIDATION_FIX.md, and update README's test count (753) once the new test files are committed.
* Download `extra_2022/` and the Fin Nifty, full-bhavcopy and Sensex raw files, then rerun the raw-data reproduction steps against the manifest snapshot.
* Fix the two steps that crash on the new data (see reports/REPRODUCTION_RERUN.md), and teach the live freeze to read extended daily files.
* Fix PR #4's four low-severity defects (reports/PR4_GEX_REVIEW.md, section 4).
* Watch the first run of the pinned CI job; bump `actions/checkout` and `actions/setup-python` to Node 24 versions.

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
PYTHONPATH=src python scripts/forward_test_p112.py count --raw data/live/forward

# Raw-data integrity
python scripts/data_manifest.py check
```

These commands write into `results/`, so run them in a copy if you need to keep the committed files untouched. `scripts/reproduce_all.py` overwrites committed results as well.
