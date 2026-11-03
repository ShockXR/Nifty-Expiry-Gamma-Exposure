# Reproducibility check: Nifty expiry gamma exposure

**Plain-language summary (for a beginner)**

1. "Reproducible" means a stranger can rebuild the stored results from the repository. Here the raw market files are not in the repository (about 25 minutes to download), so we could only check the parts that do not need them.
2. What we could check passes: the test suite (708 passed, 7 skipped that need the raw files), the list of 94 reproduction steps, and the number-tracing script (678 printed numbers, 664 match a results file, 14 are plain integers with no source, none mismatch).
3. What we could not check is marked BLOCKED, not failed: the raw-data hash check and the re-run of the real-data studies.
4. Two things deserve attention before presenting: the regenerated test ledger does not reproduce the committed document exactly (A1-A-013), and a few structural items (EGARCH gate, last-days calendar, missing minute bars) can move numbers if someone fixes them (A1-A-002 to A1-A-005).
5. Seeds are explicit (walk-forward LightGBM, bootstrap, simulator); two walk-forward runs with the same seed are bit-identical.

## Results

| Check | Result | Status |
|---|---|---|
| Baseline pytest (`OMP_NUM_THREADS=1`) | 715 tests: 708 passed, 7 skipped (real Bank Nifty and Fin Nifty tests need `data/raw`), 0 failed, 271 s. The brief's "508 passed + 4 skipped" is out of date | VERIFIED |
| Pytest after audit changes | see CODE_AUDIT.md (3 strict xfails pin known defects) | VERIFIED |
| `python scripts/reproduce_all.py --list` | 94 steps; steps 1 to 6 download data, step 7 checks hashes, the rest run studies; every step runs with `OMP_NUM_THREADS=1`, `PYTHONHASHSEED=0` and `src` first on the path | VERIFIED |
| `python scripts/data_manifest.py check` | `data/raw` absent | BLOCKED |
| Real-data studies (`reproduce_all.py` steps 8 onward) | not run | BLOCKED |
| `python scripts/trace_numbers.py --out <scratch>` | 678 claims: 664 MATCH, 14 UNTRACED, 0 mismatch; identical to the working-tree `results/number_trace/trace.csv`. At the committed head the file had 575 rows (562 MATCH, 13 UNTRACED), i.e. stale against the current documents | VERIFIED |
| The 14 untraced items | small integers in the pitch (2, 5, 20, 41, 50, 80, 100, 360 and similar), the test count 715, and 0.06 in the briefing; they have no matching results row; they were listed, not checked one by one, and none is reported as a mismatch | VERIFIED |
| `scripts/build_test_ledger.py` regenerated in a full copy | not identical to `docs/TEST_LEDGER.md` beyond the pre-specification column (A1-A-013) | PARTIALLY VERIFIED |
| Pinned versions | `requirements-tested.txt` (arch 8.0.0, lightgbm 4.7.0, numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, statsmodels 0.15.0, matplotlib 3.11.2) match the environment used for this check exactly | VERIFIED |
| Python 3.11 / 3.12 / 3.13 (CI matrix) | every `.py` parses under 3.11; the suite was executed on 3.13 only | PARTIALLY VERIFIED |
| Seeds and determinism | `WalkForwardConfig.seed=0`, LightGBM single thread; two walk-forward runs and two simulator runs with the same seed are bit-identical; `day_cluster_bootstrap` and the MCS take explicit seeds | VERIFIED |
| Non-causal numerical effect | arch EGARCH early-day variance changes by at most about 1e-4 relative in forecasts when later days change (A1-A-008) | VERIFIED |
| Undocumented manual steps | none found; the download step is explicit (`--download`) and takes 1 to 2 hours; the notebook step needs jupyter | VERIFIED |

## Hazards for the person presenting

* `reproduce_all.py` writes into `results/` and `docs/`; run it in a copy.
* The pre-specification column of the test ledger is derived from git reflog times of the original clone, so it cannot be regenerated on another clone (A1-A-013).
* Left unchanged on purpose because they would change printed numbers: the EGARCH-versus-EWMA gate that looks at later days (disclosed by the authors), the EWMA floor, the days-to-expiry label for the last days of a sample.

## How to repeat

```
cp -r <repo> /tmp/nifty_copy && cd /tmp/nifty_copy
OMP_NUM_THREADS=1 python -m pytest -q                 # about 4.5 minutes
python scripts/reproduce_all.py --list
python scripts/trace_numbers.py --out /tmp/trace_out
```
