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

## What can differ on a rerun

* The Nifty minute-bar dataset is updated daily by its maintainer, so a later download covers more days. `data/raw/minute_bars_source.json` records the commit used.
* LightGBM results can differ in the last decimal places across library versions and CPU types.
