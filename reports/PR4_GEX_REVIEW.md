# Review of PR #4: input checks in the GEX engine (false alarms and cost on real chains)

Done by an AI agent (Claude, Agent F) on 4 November 2026, about 02:26 to 02:35 UTC. I reviewed the PR #4 diff itself: `git diff origin/main origin/claude/project-thread-2lfy7y -- src/niftygex/gex.py`. I did not review agent B's later additions in the checkout. "Old" means `src/niftygex/gex.py` at `origin/main`; "new" means the same file at the PR branch. Both were loaded side by side from a scratch copy. Inputs are 20 real NSE bhavcopies (**REAL DATA**); the small edge-case probes in section 4 use made-up two-row chains (**SYNTHETIC**). I did not reclassify the 41 stress cases or rebuild the panel (agent B is doing that). Scratch code: `/tmp/claude-0/-home-user/e79a6425-7016-5765-9dd9-652f4c629e5e/scratchpad/agentF/pr4/` (`run.py`, `raw.py`, `per_day.csv`).

## Verdict: safe to merge

On 20 real days from January 2023 to November 2026:

* No valid chain raised a ValueError, so there were no false rejections.
* Every number is identical to the old engine (largest relative difference 0.0).
* The pipeline path produces no warning at all.
* The cost is about 9 seconds per 940-day panel build.

I found four defects, all low severity. None blocks the merge.

## 1. Warnings and rejections per day (REAL DATA)

**Method.** I picked 20 bhavcopies evenly spaced through `data/raw/bhavcopy` (945 files, 2023-01-02 to 2026-11-03). For each one I built the NIFTY chain the way the pipeline does (`niftygex.data.build_market_from_nse`, then `features.build_gex_panel`):

1. Take the nearest expiry strictly after the day.
2. Set the forward from put-call parity (`parity_forward`) and T = business days to expiry / 252.
3. Back out implied volatility (IV) from settlement prices with `greeks.implied_vol`.
4. Fill unsolved IVs by interpolation within each side (calls, puts).
5. Keep rows with OI > 0, and use lot size 1 (the `keep_units` setting).

**Per-day calls.** On each day I ran the panel row: `daily_summary` (with Pro-style and client-style weights; the warning counts do not depend on the weights), the near-the-money `strike_gex`, and `gamma_flip` for two rules, 41-point grid.

**Warning categories.** Only `GexInputWarning` was counted, in these kinds:

* missing IV
* IV 0
* IV below 1%
* duplicated series
* empty chain
* missing OI

| Date | Rows | IVs unsolved before the fill | ValueError | Warnings of any kind (pipeline path) | Warnings on the raw chain (no fill, OI = 0 kept) |
| --- | --- | --- | --- | --- | --- |
| 2023-01-02 | 181 | 47 | none | 0 | missing IV |
| 2023-03-14 | 147 | 26 | none | 0 | missing IV |
| 2023-05-30 | 145 | 36 | none | 0 | missing IV |
| 2023-08-09 | 138 | 18 | none | 0 | missing IV |
| 2023-10-20 | 154 | 33 | none | 0 | missing IV |
| 2024-01-04 | 144 | 29 | none | 0 | missing IV |
| 2024-03-19 | 149 | 22 | none | 0 | missing IV |
| 2024-06-04 | 191 | 32 | none | 0 | missing IV |
| 2024-08-16 | 161 | 28 | none | 0 | missing IV |
| 2024-10-28 | 227 | 59 | none | 0 | missing IV |
| 2025-01-08 | 172 | 39 | none | 0 | missing IV |
| 2025-03-21 | 244 | 37 | none | 0 | missing IV |
| 2025-06-06 | 184 | 30 | none | 0 | missing IV |
| 2025-08-14 | 176 | 23 | none | 0 | missing IV |
| 2025-10-29 | 196 | 49 | none | 0 | missing IV |
| 2026-01-09 | 165 | 24 | none | 0 | missing IV |
| 2026-03-24 | 307 | 31 | none | 0 | missing IV |
| 2026-06-10 | 198 | 32 | none | 0 | missing IV |
| 2026-08-20 | 221 | 44 | none | 0 | missing IV |
| 2026-11-03 | 186 | 37 | none | 0 | missing IV |

**What the table shows.**

* On the pipeline path, every warning count is zero on every day:
  * Unsolved IVs are filled before the engine sees them.
  * The lowest IV the solver found on any of the 20 chains was 5.4%, so the "IV below 1%" warning never fires.
  * No chain has a duplicated (strike, option type) row.
  * No chain is empty.
* On the raw chain (no fill), the only warning is "missing IV". It is correct: deep options with no time value have no IV.
* There were no false alarms and no false rejections.

**Caveat.** The pipeline never uses an expiry-day chain (it takes the nearest expiry strictly after the day). Very short-dated chains, where IV below 1% is most likely, are therefore not covered by this sample.

## 2. Cost (REAL DATA chains, one process, OMP_NUM_THREADS=1, machine shared with other agents)

| Engine | Seconds per day for the panel row (mean of 3 repeats x 20 days) |
| --- | --- |
| Old (`origin/main`) | 0.0102 |
| New (PR #4) | 0.0198 |
| Slowdown | 1.95 times |
| Extra cost for a 940-day panel build | about 9 seconds |

* About 9 seconds is negligible against the full pipeline step (6,909 seconds in `results/reproduce_log.csv`).
* Most of the extra time comes from validating the same chain again:
  * Inside `daily_summary`, `oi_walls` and `max_pain` each validate the chain once more.
  * Each `gamma_flip` call validates it again.

## 3. Logic review of `validate_chain` and the changed functions

| Point asked | Finding |
| --- | --- |
| `is_call` as numpy bool | Accepted (tested) |
| `is_call` as object dtype holding Python or numpy bools | Accepted (tested) |
| `is_call` as object dtype holding 1/0 | **Refused**, although the docstring and the error message both say 1/0 is accepted (only a numeric dtype with 1/0 passes). Defect D1 |
| `is_call` as pandas nullable boolean with a missing value | Raises pandas' own ValueError ("cannot convert to 'bool'-dtype NumPy array"), not the engine's message. Defect D2 |
| Order of NaN handling for IV | Correct. NaN IVs are not counted as negative (`iv < 0` is False for NaN) or as stale (`iv > 0` is False). IV 0 is set to NaN first. The missing-IV count then subtracts the zeros, so each row is counted once. |
| Lot size as a column instead of a scalar | Refused with "lot_size must be a positive number" (tested). The old engine silently multiplied elementwise. This is a behaviour change, but every caller in the repo passes a scalar (`market["lot_size"]`). Defect D3 (documentation) |
| Infinite IV or OI | Not rejected (tested, both pass). Infinite OI gives an infinite or NaN gross GEX. Defect D4 |
| pandas 3 copy-on-write (pandas 3.1.0 installed) | No problem. `validate_chain` works on `chain.copy()` and writes with `.loc` and whole-column assignment. It never writes through a chained index. The caller's frame is not modified, and `strike_gex` returns the caller's columns and dtypes. |
| Can callers filter the warnings? | Yes. `GexInputWarning` is a public subclass of `UserWarning`, so `warnings.filterwarnings("ignore", category=gex.GexInputWarning)` or `"error"` works. Note: `stacklevel=3` points at the caller's line, so with Python's default filter a warning raised inside a 940-day loop is shown only once per line. Use the "always" filter to count warnings per day. |
| Numbers on valid input | Identical to the old engine on all 20 real days (largest relative difference 0.0, no NaN mismatches) |

## 4. Defects

| ID | Severity | Defect | Suggested fix |
| --- | --- | --- | --- |
| D1 | Low | Object-dtype `is_call` holding 1/0 is refused although the docs say 1/0 is accepted | Map object values in {0, 1, True, False} explicitly, or say "numeric 1/0" in the docs |
| D2 | Low | Nullable boolean with a missing value gets pandas' error message, not the engine's | Check `col.isna().any()` first and raise the engine's message |
| D3 | Low (documentation) | A per-row lot size is now refused (it used to be broadcast) | State in the docstring that `lot_size` must be a scalar |
| D4 | Low | Infinite IV or OI is not rejected | Add `np.isfinite` to the OI and IV checks (allowing NaN) |
| (cost) | Informational | The chain is validated 4 or more times per panel row | Optional: pass the checked copy to `oi_walls`, `max_pain` and `gamma_flip` |

## 5. Is PR #4 safe to merge?

Yes. On real chains it does not change any number, it rejects nothing valid, and it adds no warnings on the pipeline path. It costs about 9 seconds per panel build. D1 to D4 can be fixed in a follow-up.
