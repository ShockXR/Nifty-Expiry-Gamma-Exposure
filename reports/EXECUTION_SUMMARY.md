# Execution summary and restart checkpoint (niftygex)

3 November 2026. Four AI agents (Sonnet) and a coordinator worked on this repository. A container restart stopped two agents partway; their files were on disk and both agents were resumed and finished. Nothing was lost.

## What was completed

| Workstream | Output | State |
|---|---|---|
| Code audit | `CODE_AUDIT.md`, `REPRODUCIBILITY.md`, `tests/test_audit_agent1.py` (38 tests), `findings/A1_A.csv` (14) | complete |
| Documents and methodology | `METHODOLOGY_VALIDATION.md`, `PROFESSOR_DEFENSE.md`, `checks/equation_vs_code_check.py`, `findings/A2_A.csv` (11); README, pitch and doc wording fixes | complete |
| Figures and demo | `VISUAL_QA.md`, `DEMO_WALKTHROUGH.md`, `screenshots/`, `figure_regen/`, 62 figures redrawn, `findings/A3_A.csv` (13) | complete except the BLOCKED diurnal-curve figures |
| Backtest and stress | `BACKTEST_STRESS_REPORT.md`, `RISK_REGISTER.md`, `stress/`, `findings/A4_A.csv` | A1 to A4 done, A5 BLOCKED |
| Coordinator | `MASTER_BACKTEST_REPORT.md`, this file, test count updates, regression runs | complete |

## Tests

| Run | Result |
|---|---|
| Before any edit | 715 collected: 708 passed, 7 skipped (need `data/raw`), 0 failed, 271 s |
| After all agents | 743 passed, 7 skipped, 3 expected failures (strict xfail tests that pin known defects), 0 failed |
| Number tracer | 678 claims: 664 MATCH, 14 UNTRACED, 0 MISMATCH, 0 pitch numbers uncovered |
| Not run | raw-data steps of `reproduce_all.py` (94 steps), `data_manifest.py check`, perturbation rerun A5 |

## Changes made to code, documents and figures

Code: Diebold-Mariano test handles empty input; docstring corrections; plot labels (REAL or SYNTHETIC, day count, development or hold-out note); `run_pipeline.py` passes the data label; `build_test_ledger.py` wording matches the document. Documents: test count 753, "same information" wording removed, event-study caveat added, pitch sentence on number checking now states its exceptions, README says five sign rules are coded and three used. No printed result number changed.

## Verified, unverified, failed

VERIFIED (OUR CHECK): headline gains and DM statistics recomputed independently; option maths; walk-forward feature causality; demo numbers and pages.
PARTIALLY VERIFIED: last-hour gain (weak), figures (sampled), data integrity (minute bars only).
FAILED: look-ahead in the EGARCH versus EWMA choice (impact on the headline unmeasured); GEX engine accepts invalid input silently.
BLOCKED: A5 perturbation rerun; full raw-data comparison; five diurnal-curve figures.

## Follow-up on 4 November 2026 (after the merge of PRs 1 to 3)

Next step 2 below is done. The GEX engine (`src/niftygex/gex.py`) now checks its input with `validate_chain`, used by `strike_gex`, `net_gex_at`, `gamma_flip`, `oi_walls`, `max_pain` and `daily_summary`:

* It raises `ValueError` for input with no sensible meaning: a missing column (the message names the schema), option types given as strings such as "CE"/"PE" (before, every string counted as a call), negative open interest, negative IV, a nonpositive strike, a spot, lot size or time to expiry that is not a positive finite number (T = 0 used to give NaN gamma and a net GEX of exactly 0.0), and participant weights outside [-1, 1].
* It warns (`GexInputWarning`) for input that can be real but changes the answer: missing open interest or IV (with the count), IV of exactly 0 (treated as missing), IV below 1 percent (stale quote), duplicated series and an empty chain.
* Totals over rows that are all missing, and `daily_summary` on an empty chain, are NaN instead of 0.0. `oi_walls` and `max_pain` return NaN when there is no positive open interest (before, they returned the first strike) and `max_pain` no longer crashes on an empty chain.
* The engine still cannot check units (open interest in shares instead of contracts, A3-21 and A3-22): that remains documented, not detected.

Evidence that no committed GEX panel changes (OUR CHECK): none of the six committed `daily_gex_panel.csv` files has a net or gross GEX of exactly 0 or NaN, a T or spot at or below 0, or a day with no open interest, which are the only cases whose output changed; every experiment floors T at one trading day; and the synthetic panel rebuilt from seed 7 with the new engine is byte-identical to the one built with the old engine. Tests: `tests/test_gex_input_checks.py` (34 new tests, one per A3 case); one older test that pinned the T = 0 behaviour was updated. Full suite 777 passed, 7 skipped, 3 xfailed (also with the pinned versions in `requirements-tested.txt`); `scripts/trace_numbers.py` 664 MATCH, 0 MISMATCH. A fuzz run of 3,000 random corrupted chains gave an error, a warning or a finite number every time, never a silent 0 or NaN.

## Next steps, in order

1. Download the raw data (`docs/DATA_ACQUISITION.md`, about 25 minutes), make the EGARCH choice and EWMA floor causal, rerun the Nifty pipeline and report the change in the headline gain, then rerun `scripts/trace_numbers.py` and update any quoted number. Add a test that replaces (not removes) the future across a refit block.
2. Done on 4 November 2026 (see the follow-up above): input validation in the GEX engine.
3. Check the six overflowing deck slides in the real viewer.
4. Run the perturbation experiments (A5) and P120 (registered, not run).

## How to resume

```bash
git checkout main
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-tested.txt pytest && pip install -e . --no-deps
OMP_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider      # expect 830 passed, 7 skipped, 3 xfailed (834 passed, 3 skipped with data/raw)
python scripts/trace_numbers.py                                 # expect 664 MATCH, 0 MISMATCH
```

Hazard: running `scripts/reproduce_all.py` in place overwrites committed result files. Run it in a copy.
