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

Result: installation took about 27 seconds and all 34 tests passed in about 9 seconds (Python 3.13.18, Linux). The exact package versions are in `requirements-tested.txt`. The tests need no market data: they run on small synthetic files written in NSE's layouts.

The same tests run automatically on GitHub for Python 3.11, 3.12 and 3.13 on every push (`.github/workflows/tests.yml`).

## Rebuilding the real-data results

Raw data are not in the repository. `docs/DATA_ACQUISITION.md` lists the four commands that download them from free sources and rerun the pipeline (about 25 minutes of downloading plus the model runs). Every script that produces a result writes its configuration next to the output (for example `results/pipeline_nse/run_info.json`), and random seeds are fixed.

## One command, with data hashes

```bash
python scripts/reproduce_all.py --download   # downloads, checks hashes, reruns every study
python scripts/reproduce_all.py              # if data/raw is already there
```

`data/MANIFEST.csv` (committed) lists every raw file with its SHA-256 hash (of the decompressed content for `.gz` files, because gzip headers carry a time stamp). `scripts/data_manifest.py check` compares data/raw with it, and `reproduce_all.py` stops if anything is missing or different. The minute-bar datasets are fetched at the exact git commits recorded in `data/raw/minute_bars_source.json` and cut at 2 November 2026 (`--nifty-ref`, `--bank-ref`, `--until`), so a later download gives the same files. Each step's run time and exit status go to `results/reproduce_log.csv`.

Check on 2 November 2026: rerunning `real_day_level`, `momentum_gex` and `closing_auction_check` from the hashed data reproduced their committed outputs byte for byte. A full run is logged here once the remaining studies are added.

Files for the live dry run (`experiments/live_dry_run.py score`) go to `data/live/`, not `data/raw/`, so scoring a live day never changes the hashed data.

## What can differ on a rerun

* The Nifty minute-bar dataset is updated daily by its maintainer. Use the pinned commit and `--until 2026-11-02` (as `reproduce_all.py` does) to get the same file.
* NSE could in principle revise an archived file; the manifest check would then report it.
* LightGBM results can differ in the last decimal places across library versions and CPU types.
