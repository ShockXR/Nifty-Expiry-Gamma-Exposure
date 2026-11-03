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
python scripts/reproduce_all.py --download   # downloads, checks hashes, reruns every study
python scripts/reproduce_all.py              # if data/raw is already there
```

`data/MANIFEST.csv` (committed) lists every raw file with its SHA-256 hash (of the decompressed content for `.gz` files, because gzip headers carry a time stamp). `scripts/data_manifest.py check` compares data/raw with it, and `reproduce_all.py` stops if anything is missing or different. The minute-bar datasets are fetched at the exact git commits recorded in `data/raw/minute_bars_source.json` and cut at 2 November 2026 (`--nifty-ref`, `--bank-ref`, `--until`), so a later download gives the same files. Each step's run time and exit status go to `results/reproduce_log.csv`.

Check on 2 November 2026: rerunning `real_day_level`, `momentum_gex` and `closing_auction_check` from the hashed data reproduced their committed outputs byte for byte. Full run on 3 November 2026 (`results/reproduce_log.csv`): the 22 steps that `reproduce_all.py` listed at the time, from the hash check to the Sensex swap study, took about 2 hours 20 minutes on one core, of which the five-seed forecasting pipeline took 1 hour 55 minutes. Every step finished without error except the notebook, which stopped on a model name that had changed and was fixed and re-executed (commit eb2127d). What the comparison with the committed outputs showed:

* The headline forecasts (`results/pipeline_nse_v2/`, the model study behind REAL_DATA_RESULTS.md finding 2) came out byte for byte the same.
* The other studies moved only in late decimal places, because they now read the GEX panels written by the second forecasting run rather than the first. The cells quoted in the documents that moved were updated (for example the R5 table in REAL_DATA_RESULTS.md and the F2_ECONOMICS deviation note); the pinning-mechanism study (P87) was identical.
* A few committed diagnostic files had come from the first forecasting run and were overwritten by second-run versions. The first-run files, which some documents cite as history, are kept in `results/archive_v1_first_run/`.

The steps added to `reproduce_all.py` afterwards (the synthetic control P96, the Bank Nifty replication P91, the referee-response analyses, the DSR check, the test ledger and the number trace) were each run separately on the same hashed data when they were written; they were not part of this timed run. The Bank Nifty pipeline adds about 80 minutes.

Files for the live dry run (`experiments/live_dry_run.py score`) go to `data/live/`, not `data/raw/`, so scoring a live day never changes the hashed data.

Forward test (data from 3 November 2026 on): pre-registered in `docs/prereg/FORWARD_TEST.md`, run with `python scripts/forward_test.py --start 2026-11-03 --end END --raw data/live/forward` at the commit that adds that file (it reports only counts of days until each stopping point); `results/forward_test/dryrun_2025-01-01_2026-07-31/` is a software check on past data, not the forward test.

## What can differ on a rerun

* The Nifty minute-bar dataset is updated daily by its maintainer. Use the pinned commit and `--until 2026-11-02` (as `reproduce_all.py` does) to get the same file.
* NSE could in principle revise an archived file; the manifest check would then report it.
* LightGBM results can differ in the last decimal places across library versions and CPU types.
