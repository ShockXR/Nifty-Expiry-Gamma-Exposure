# Reproducibility checks

## Clean install (2 November 2026)

To make sure the repository works for someone who clones it fresh, we cloned the branch into a new folder, created an empty virtual environment, installed the package the way a reader would, and ran the tests:

```bash
git clone https://github.com/ShockXR/Nifty-Expiry-Gamma-Exposure.git
cd Nifty-Expiry-Gamma-Exposure
python -m venv .venv && source .venv/bin/activate
pip install ".[download]" pytest
OMP_NUM_THREADS=1 python -m pytest -q
```

Result: installation took about 27 seconds and all 34 tests passed in about 9 seconds (Python 3.13.18, Linux). The exact package versions are in `requirements-tested.txt`. The committed real-data results were produced in an environment with slightly different patch versions (numpy 2.5.3, pandas 3.0.5); those are listed in `requirements-results.txt`. The tests need no market data: they run on small synthetic files written in NSE's layouts.

The same tests run automatically on GitHub for Python 3.11, 3.12 and 3.13 on every push (`.github/workflows/tests.yml`).

**Repeated on 3 November 2026** with a fresh clone of the pushed branch at commit a3a84c7: `pip install ".[download]" pytest` took 40 seconds and resolved numpy 2.5.4, pandas 3.1.0, scipy 1.18.1 and LightGBM 4.7.0 (the versions in `requirements-tested.txt`); the suite then gave 508 passed and 4 skipped in 82 seconds (the 4 skipped tests need the raw Bank Nifty data, which a fresh clone does not have). Neither the tree nor the git history contains a PDF, the private working paper or any credential. pandas 3.1 warns that `DatetimeIndex.weekday`, which the code uses in many places, will be removed in pandas 4, so `pyproject.toml` now asks for pandas below 4.

## Rebuilding the real-data results

Raw data are not in the repository. `docs/DATA_ACQUISITION.md` lists the four commands that download them from free sources and rerun the pipeline (about 25 minutes of downloading plus the model runs). Every script that produces a result writes its configuration next to the output (for example `results/pipeline_nse/run_info.json`), and random seeds are fixed.

## One command, with data hashes

```bash
python scripts/reproduce_all.py --download   # downloads, checks hashes, reruns every step it lists
python scripts/reproduce_all.py              # if data/raw is already there
python scripts/reproduce_all.py --list       # prints every step and its command; runs nothing
```

`data/MANIFEST.csv` (committed) lists every raw file with its SHA-256 hash (of the decompressed content for `.gz` files, because gzip headers carry a time stamp). `scripts/data_manifest.py check` compares data/raw with it, and `reproduce_all.py` stops if anything is missing or different. The minute-bar datasets are fetched at the exact git commits recorded in `data/raw/minute_bars_source.json` and cut at 2 November 2026 (`--nifty-ref`, `--bank-ref`, `--until`), so a later download gives the same files. Each step's run time and exit status go to `results/reproduce_log.csv`.

Check on 2 November 2026: rerunning `real_day_level`, `momentum_gex` and `closing_auction_check` from the hashed data reproduced their committed outputs byte for byte. Full run on 3 November 2026 (`results/reproduce_log.csv`): the 22 steps that `reproduce_all.py` listed at the time, from the hash check to the Sensex swap study, took about 2 hours 20 minutes on one core, of which the five-seed forecasting pipeline took 1 hour 55 minutes. Every step finished without error except the notebook, which stopped on a model name that had changed and was fixed and re-executed (commit eb2127d). What the comparison with the committed outputs showed:

* The headline forecasts (`results/pipeline_nse_v2/`, the model study behind REAL_DATA_RESULTS.md finding 2) came out byte for byte the same.
* The other studies moved only in late decimal places, because they now read the GEX panels written by the second forecasting run rather than the first. The cells quoted in the documents that moved were updated (for example the R5 table in REAL_DATA_RESULTS.md and the F2_ECONOMICS deviation note); the pinning-mechanism study (P87) was identical.
* A few committed diagnostic files had come from the first forecasting run and were overwritten by second-run versions. The first-run files, which some documents cite as history, are kept in `results/archive_v1_first_run/`.

### Which steps have been timed end to end (updated 3 November 2026)

Since this update `reproduce_all.py` lists a step for every registered study in `docs/prereg/` except the two forward tests (`FORWARD_TEST.md` and P112, which need data after 2 November 2026 and run with `scripts/forward_test.py`) P117's six steps and P118's five were added when those studies finished (groups 4 and 5 below). `python scripts/reproduce_all.py --list` prints all 85 study steps with their commands. Only the first group below has been run end to end in one go. Every step outside it was run on its own, on the same hashed data, when its study was done; the run times below are the ones stated in each step's document, not a new measurement.

1. **Timed together on 3 November 2026 (about 2 hours 20 minutes on one core; `results/reproduce_log.csv`, 8,529 seconds in all):** the hash check, `validate_minute_bars`, `check_oi_units`, `run_pipeline` (6,909 seconds), `forecast_diagnostics`, `real_day_level`, `momentum_gex`, `closing_auction_check`, `replicate_paper`, `pinning_study`, `notebook`, `falsification`, `intraday_gamma`, `dealer_sign`, `expiry_event_study`, `sensex_confound`, `connectedness_gamma`, `power_real`, `rekey_check`, `pinning_mechanism`, `f2_economics` and `sensex_swap` (22 steps).
2. **Added earlier on 3 November 2026, after the timed run; not yet timed end to end.** Stated run times: `synthetic_control` (P96) about 1 minute (`docs/SYNTHETIC_CONTROL.md`); `banknifty_pipeline` (P91) 78 minutes (`docs/BANKNIFTY_FORECAST.md`); `finnifty_pipeline` (P104) 61 minutes on one thread (`docs/THIRD_INDEX_FORECAST.md`); `referee_forecast` 193 seconds (`docs/REFEREE_RESPONSE_FORECAST.md`); `referee_f2` about 3 minutes and `referee_f2_positions` about 75 seconds (`docs/REFEREE_RESPONSE_F2.md`); `referee_expiry` about 4 minutes (`docs/REFEREE_RESPONSE_EXPIRY.md`); `p103_economic_value` (P103) about 5 seconds (`docs/P103_ECONOMIC_VALUE.md`). No time is stated for `p91_banknifty_summary`, `validate_finnifty_bars`, `p104_finnifty_summary`, `p104_finnifty_exploratory`, `referee_forecast_gw`, `f2_dsr_power`, `test_ledger` or `number_trace`. Stated total: about 2 hours 32 minutes.
3. **Added by this update; not yet timed end to end.** Each study's commands are copied from its document (for P106 and P110, from the plan that the document points to), in dependency order, after `p103_economic_value` and before the test ledger and the number trace:

   | Study | Steps | Stated run time (document) |
   |---|---|---|
   | P99 seed bootstrap | `run`, `analyze`, `context` | 71 minutes with 2 worker processes (`docs/SEED_BOOTSTRAP.md`) |
   | P106 gain decomposition | `run` for Nifty, Bank Nifty and Fin Nifty, `analyze` | about 1 hour 45 minutes on 2 cores (`docs/P106_GAIN_DECOMPOSITION.md`, Section 10) |
   | P107 pooled test | `count`, `run`, post hoc | about 15 seconds on one core (`docs/P107_POOLED_THREE_INDEX.md`, Section 12) |
   | P108 blind window | `search`, `posthoc`, `phase3_compare` | about three minutes (`docs/P108_BLIND_WINDOW.md`, Section 9) |
   | P109 mcsGARCH extensions | `run` for three indices, `analyze` | about 16 minutes on 2 cores (`docs/P109_MCSGARCH_EXTENSIONS.md`, Section 17) |
   | P110 stress test | `run` for Bank Nifty and Fin Nifty, `analyze` | about 2.5 hours on one core (`docs/P110_GAIN_STRESS_TEST.md`, Section 11) |
   | P113 connectedness settings | one command | about 2.5 minutes, one process (`docs/P113_CONNECTEDNESS_SETTINGS.md`, Section 8) |
   | P114 persistence, fixed model | `inputs`, `fit`, `lr`, `rolling`, `boot`, `analyze`, post hoc | about 36 minutes on one core (`docs/P114_PERSISTENCE_FIXED_MODEL.md`, Section 15) |
   | P116 benchmark as printed | `run` for three indices, `analyze`, post hoc | about 8 minutes on 2 cores (`docs/P116_BENCHMARK_AS_PRINTED.md`, Section 16) |
   | P111 demo | `--pick` and the three demo days | a few seconds per command (`docs/DEMO.md`, Section 2) |

   Stated total: about 6 hours 32 minutes. P99, P106, P109 and P116 run with 2 worker processes, as their documents' commands do; their jobs' own computing times add up to 8,026, 11,703.5, 1,229.3 and 552.4 seconds (from the studies' `timing.csv` files as quoted in their documents; for P109 and P116 the sum of the three per-index times).

4. **Added when P117 finished (3 November 2026, evening).** `p117_inputs`, `p117_windows_nifty`, `p117_windows_banknifty`, `p117_control`, `p117_analyze` and `p117_posthoc_flat_days`, after the demo and before the test ledger: about 61 minutes on one core in the study's own run (`docs/P117_PERSISTENCE_PLACEBO.md`, Section 10).
5. **Added when P118 finished (3 November 2026, evening).** `p118_repro_nifty`, `p118_run_nifty`, `p118_run_banknifty`, `p118_run_finnifty` and `p118_analyze`, after P117 and before the test ledger: about 1 hour on 2 cores in the study's own run (`docs/P118_HYBRID_ON_PAPER_MCSGARCH.md`, Section 16). They need P116's block files, which the P116 steps write.

So the stated run time of the whole list is about 13.4 hours (2 hours 20 minutes timed, plus about 2 hours 32 minutes, about 6 hours 32 minutes, about 1 hour and about 1 hour stated), plus the eight steps of group 2 whose documents state no time. Each script and option the new steps name was checked without running any study (`--help` of every script and subcommand that takes options, a compile check of the two post hoc scripts that take none, and the eight plan commits looked up in git); `tests/test_reproduce_all.py` checks that every registered study has a step, that every script and option a step names exists, and that `--list` runs nothing. Since this update `reproduce_all.py` also puts the repository's `src/` first on `PYTHONPATH`, as the study documents' commands do.

Files for the live dry run (`experiments/live_dry_run.py score`) go to `data/live/`, not `data/raw/`, so scoring a live day never changes the hashed data.

Forward test (data from 3 November 2026 on): pre-registered in `docs/prereg/FORWARD_TEST.md`, run with `python scripts/forward_test.py --start 2026-11-03 --end END --raw data/live/forward` at the commit that adds that file (it reports only counts of days until each stopping point); `results/forward_test/dryrun_2025-01-01_2026-07-31/` is a software check on past data, not the forward test.

## What can differ on a rerun

* The Nifty minute-bar dataset is updated daily by its maintainer. Use the pinned commit and `--until 2026-11-02` (as `reproduce_all.py` does) to get the same file.
* NSE could in principle revise an archived file; the manifest check would then report it.
* LightGBM results can differ in the last decimal places across library versions and CPU types.
