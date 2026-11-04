# GEX engine input validation: review of PR #4 plus small additions (findings A4-A-002 and A4-A-003)

Written by an AI agent (Agent B, Claude) on 2026-11-04. No result number in any other document was changed, nothing under `results/` was written, and the headline pipeline was not rerun.

## In one paragraph

The GEX engine (`src/niftygex/gex.py`) used to accept broken input without complaint: a failed download, a zero time to expiry or a zero implied volatility all gave a net GEX of exactly 0.0, which looks like a calm "neutral" day. PR #4 (branch `claude/project-thread-2lfy7y`, commit 08aee15, written by another AI thread) adds input checks. I reviewed it against the 41 SYNTHETIC stress cases of `reports/stress/a3_gex_stress.py`, found four small gaps, fixed them on top of PR #4 (every change is marked "Agent B" in the code), and rebuilt the committed REAL DATA GEX panels from the raw NSE files with PR #4's code to check that no committed number moves.

Words used below. DETECTED: the engine raises its own clear error, returns a visible NaN ("not a number", meaning unknown) or warns with `GexInputWarning`. SAFELY HANDLED: a sensible, correct result. SILENTLY IGNORED: accepted with a wrong or misleading number and no signal. CRASHED: a failure from library internals with no useful message.

## 1. What PR #4 does (review)

`validate_chain` runs at the start of every public function and returns a cleaned copy (the caller's frame is not changed).

* `ValueError` for: missing columns (message lists what is needed and what is there); option type that is not True/False or 1/0, including the strings CE/PE (PR #4 refuses them rather than mapping them; both choices end the old bug where every string counted as a call); negative open interest (OI); negative implied volatility (IV); strike, spot, lot size, T or forward at or below 0 or not finite; participant weights outside [-1, 1].
* `GexInputWarning` for input that is possible but changes the answer: missing OI or IV (with the count of rows left out), IV of exactly 0 (treated as missing), IV below 1% (stale quote), duplicated (strike, option type) rows, an empty chain.
* `total()` sums like pandas but gives NaN when every value is missing, so an all-NaN chain is no longer 0.0. Walls and max pain are NaN when there is no open interest; `max_pain` no longer crashes on an empty chain.

My assessment: correct and well scoped. On valid input it keeps the arithmetic (section 4 checks this on REAL DATA). The design choice that matters most for users: several problems only warn, so a caller who ignores Python warnings still gets a number (for partial NaN, stale IV, duplicates); the full-NaN and empty cases do give NaN.

## 2. Gaps I found in PR #4 and fixed on top of it (marked "Agent B" in `src/niftygex/gex.py`)

1. `daily_summary` used a plain `.sum()` for `gross_gex_atm1pct`, so a chain with all OI missing still gave 0.0 in that column. Now uses `total()`. On any chain with at least one valid row the number is the same.
2. No upper bound on IV: an IV of 6 (600%) or infinity was accepted (infinity gave a silent NaN). Now `ValueError` above `IV_MAX = 5.0`, the top of the bracket of `greeks.implied_vol`, so every IV that solver can return still passes (A3-15, IV 500%, is still SAFELY HANDLED).
3. Infinite open interest gave a silent NaN. Now `ValueError`.
4. Coverage: a chain with calls only or puts only (A3-26), or with no strike within 2% of spot (A3-25), gave a confident but wrong net GEX with no signal. `daily_summary` now warns for both. These are warnings, not errors, because a legitimate subset can look like this.

Not fixed (out of scope, noted for later): `gex_by_strike` assumes a boolean `is_call` (it is always fed `strike_gex` output, which is boolean); `features.build_gex_panel` sums the near-the-money columns (`gross_gex_nm`, `net_gex_*_nm`) itself with pandas, so they would read 0.0 if a whole near-the-money subset were missing. The pipeline drops rows with OI at or below 0 and fills IV gaps before calling the engine, which is why no committed day has this.

## 3. The 41 stress cases (SYNTHETIC chains)

Rerun of `reports/stress/a3_gex_stress.py` against each version (output written to scratch space; the committed `reports/stress/*.csv` files are untouched), plus a probe of the engine's own aggregates (`daily_summary`, `net_gex_at`), because the stress script sums `strike_gex` rows with pandas itself.

| Class | Before (committed engine) | PR #4 | PR #4 plus Agent B fixes | Agent B's own first version (not merged, for comparison) |
|---|---|---|---|---|
| DETECTED | 3 | 23 | 25 | 24 |
| SAFELY HANDLED | 11 | 14 | 14 | 15 |
| SILENTLY IGNORED | 24 | 4 | 2 | 2 |
| CRASHED | 3 | 0 | 0 | 0 |

Cases that changed from the committed engine:

| Case | Before | PR #4 | PR #4 plus Agent B |
|---|---|---|---|
| A3-02 all OI zero | SILENTLY IGNORED | SAFELY HANDLED (walls, max pain NaN) | same |
| A3-03 negative OI | SILENTLY IGNORED | DETECTED (error) | same |
| A3-04 20% NaN OI | SILENTLY IGNORED | DETECTED (warning, 39 of 194 rows) | same |
| A3-05 all OI NaN | SILENTLY IGNORED | DETECTED (NaN, but `gross_gex_atm1pct` still 0.0) | DETECTED (all NaN) |
| A3-06 all IV NaN | SILENTLY IGNORED | DETECTED (NaN) | same |
| A3-07 25% NaN IV | SILENTLY IGNORED | DETECTED (warning) | same |
| A3-08 empty chain | SILENTLY IGNORED | DETECTED (warning, NaN) | same |
| A3-09, A3-10 T = 0, T < 0 | SILENTLY IGNORED | DETECTED (error) | same |
| A3-13 IV = 0 | SILENTLY IGNORED | DETECTED (warning, NaN) | same |
| A3-14 IV negative | SILENTLY IGNORED | DETECTED (error) | same |
| A3-16 IV 0.0001 | SILENTLY IGNORED | DETECTED (warning) | same |
| A3-17 to A3-20 spot or lot size at or below 0 | SILENTLY IGNORED | DETECTED (error) | same |
| A3-25 no strike within 2% of spot | SILENTLY IGNORED | SILENTLY IGNORED | DETECTED (warning) |
| A3-26 calls only | SILENTLY IGNORED | SILENTLY IGNORED | DETECTED (warning) |
| A3-27 duplicated rows | SILENTLY IGNORED | DETECTED (warning) | same |
| A3-28 option type CE/PE | SILENTLY IGNORED | DETECTED (error) | same |
| A3-32 weights (2, -3) | SILENTLY IGNORED | DETECTED (error) | same |
| A3-33, A3-34 missing or renamed column | CRASHED | DETECTED (error naming the column) | same |
| A3-37 daily_summary on zero-OI chain | SILENTLY IGNORED | SAFELY HANDLED | same |
| A3-38 oi_walls on empty chain | SAFELY HANDLED | SAFELY HANDLED (NaN plus warning) | same |
| A3-39 max_pain on empty chain | CRASHED | SAFELY HANDLED (NaN) | same |

Still SILENTLY IGNORED in every version: A3-21 (OI in shares instead of contracts, 75 times too large) and A3-22 (lot size 1 instead of 75). The engine cannot tell a unit mistake from a valid chain; risk R-A5 and `scripts/check_oi_units.py` cover it. `tests/test_gex_validation.py` pins this limitation so a future fix shows up. A3-41 (the IV solver returns its cap 5.0 for a price just above intrinsic value) is unchanged; the solver was not touched.

Where my own first version differed from PR #4: it mapped CE/PE to booleans instead of refusing them, and it raised an error for an empty chain and for IV = 0 where PR #4 warns and returns NaN. Both are defensible; I kept PR #4's choices to avoid two conflicting fixes. My first version is kept in scratch space only.

## 4. Proof that no committed GEX number changed (REAL DATA)

Method: the daily GEX panel was rebuilt from the raw NSE files in `data/raw` (bhavcopies, participant open interest, India VIX and the official trading-day list downloaded in this session, minute bars from the sources in `data/raw/minute_bars_source.json`) with the pipeline's own functions, `scripts/run_pipeline.load_market` and then `niftygex.features.build_gex_panel`, and the committed runs' arguments (source nse, oi_unit keep_units, rate 0.065, lot_size null, min_coverage 0.95, start 2023-01-01; end 2026-11-02 for Nifty, 2026-04-22 for Bank Nifty, as in each `run_info.json`). Output went to scratch space, never to `results/`. Each panel was written with the same `DataFrame.to_csv` call as the pipeline and compared with the committed file column by column (maximum absolute and relative difference, NaN pattern) and byte by byte.

| Engine version | Panel (REAL DATA) | Rows rebuilt / committed | Numeric columns compared | Max absolute difference | Max relative difference | NaN pattern | Whole file |
|---|---|---|---|---|---|---|---|
| committed engine (HEAD before PR #4) | `results/pipeline_nse_v2/daily_gex_panel.csv` (Nifty) | 942 / 942 | 19 | 0 | 0 | identical | byte-identical |
| committed engine | `results/pipeline_nse_banknifty/daily_gex_panel.csv` (Bank Nifty) | 810 / 810 | 19 | 0 | 0 | identical | byte-identical |
| PR #4 (commit 08aee15) | Nifty | 942 / 942 | 19 | 0 | 0 | identical | byte-identical |
| PR #4 | Bank Nifty | 810 / 810 | 19 | 0 | 0 | identical | byte-identical |
| PR #4 plus Agent B fixes | Nifty | 942 / 942 | 19 | 0 | 0 | identical | byte-identical |
| PR #4 plus Agent B fixes | Bank Nifty | 810 / 810 | 19 | 0 | 0 | identical | byte-identical |

No `GexInputWarning` was raised on any of the 942 Nifty or 810 Bank Nifty days, so the real chains trip none of the new checks. The first row matters as a control: it shows the rebuild itself reproduces the committed files exactly, so a zero difference for PR #4 means the engine change, not luck, kept the numbers. VERIFIED: no committed GEX number changes under PR #4 or under PR #4 plus the Agent B fixes. Only the daily GEX panels were rebuilt; downstream files (forecasts, tests of the forecasts) were not rerun, but they read these panels, which are unchanged.

## 5. Tests

`tests/test_gex_validation.py` (written by Agent B, 31 tests, SYNTHETIC chains): the four gaps of section 2, a clean-chain test (no warning, same number), one test per originally silent stress case checking that the engine now gives a signal (22 cases), and 2 tests that pin the unit-mistake limitation. It does not repeat what PR #4's `tests/test_gex_input_checks.py` already tests. `tests/test_gex.py`, `tests/test_gex_input_checks.py` and `tests/test_gex_validation.py` together: 70 passed.

The three strict xfail tests in `tests/test_audit_agent1.py` (A1-A-002, A1-A-003, A1-A-004) are about the expiry calendar and the EGARCH/EWMA fallback, not the GEX engine, so they were left as they are. `tests/test_audit_agent1.py` itself is PR #4's version.

Full suite, `OMP_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider`: before (as given by the coordinator) 743 passed, 7 skipped, 3 xfailed; after (PR #4 plus the Agent B fixes, run at 02:33 UTC with the other agents' new test files present): 828 passed, 3 skipped, 3 xfailed, 0 failed (495 s). The extra passes come from PR #4's and this file's tests and from other agents' new test files; 4 fewer skips, probably because the raw NSE data are now in data/raw (not checked).

## 6. Status per check

| Check | Status |
|---|---|
| PR #4 code review against the 41 SYNTHETIC stress cases | VERIFIED (23 DETECTED, 14 SAFELY HANDLED, 4 SILENTLY IGNORED, 0 CRASHED) |
| Agent B fixes on top of PR #4 (IV cap, infinite OI, all-NaN near-the-money total, coverage warnings) | VERIFIED by tests (2 silent cases left, both unit mistakes) |
| REAL DATA Nifty and Bank Nifty daily GEX panels unchanged (old engine, PR #4, PR #4 plus fixes) | VERIFIED (byte-identical, 942 and 810 days) |
| GEX test files (`test_gex.py`, `test_gex_input_checks.py`, `test_gex_validation.py`) | VERIFIED (70 passed) |
| Full test suite after the change | VERIFIED (828 passed, 3 skipped, 3 xfailed, 0 failed) |
| Downstream results (forecasts, Diebold-Mariano tests) rebuilt from the panels | NOT RUN (not needed: their GEX inputs are byte-identical) |
| Unit mistakes (OI in shares, wrong lot size) detected by the engine | FAILED by design: the engine cannot see them; covered by R-A5 and `scripts/check_oi_units.py` |
| Fin Nifty GEX panel comparison | NOT RUN (not asked; its raw inputs were not checked) |

Files: `src/niftygex/gex.py` (PR #4 plus marked Agent B lines), `tests/test_gex_validation.py`, this report, and the status cell of R-A4 in `reports/RISK_REGISTER.md`. Scratch evidence (not committed): the rebuilt panels, comparison tables and both raw stress outputs are in the agent's scratch folder.
