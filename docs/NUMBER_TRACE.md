# Number trace (P98): does every printed number come from a results file?

A referee asked that every number in the write-ups be traceable to a results file. This note explains the
check built for that, how to run it, and what it found. Nothing in the documents was edited; the mismatches
below are reported for the author to fix.

## What the check does, in plain words

The project's documents (the pitch, the README, the briefing, the study write-ups and the private working
paper) quote many numbers: "5.8% better", "p = 0.0006", "21,239 returns". Each of them was copied by hand
from a CSV or JSON file that a script wrote into `results/`. Copying by hand goes wrong in quiet ways: a
number is rounded the wrong way, a table is not updated after a rerun, or a sentence keeps an old value.

The check works from a list, the **registry** (`docs/number_registry.csv`). Each row is one printed number
and says:

| Column | Meaning | Example |
| --- | --- | --- |
| `claim_id` | a short name for the row | `PI13` |
| `document` | where the number is printed (`PAPER` means the private working paper) | `PITCH_PANDEY.md` |
| `quoted_text` | a short piece of the sentence around it | `the last hour 5.8%, 7.6% and 6.4% better ...` |
| `value` | the number exactly as printed | `5.8` |
| `rounding_or_tolerance` | how to compare (empty means "must round to the printed digits") | `neg` |
| `results_file` | the file the number comes from | `results/pipeline_nse_v2/scoreboard_next15.csv` |
| `selector` | which cell of that file | `csv:group=all&model=hybrid_base@qlike_vs_mcsgarch_pct` |
| `note` | anything a reader should know | |

For every row, `scripts/trace_numbers.py` asks two questions:

1. **Is the quoted text still in the document, with the printed number inside it?** Line breaks and extra
   spaces do not matter. If someone edits the sentence, this fails, so the registry cannot silently go
   stale.
2. **Does the number in the results file round to the printed number?** The file value is read with the
   selector, optionally transformed (see below), and compared at the precision of the printed number.

Each row then gets one status:

| Status | Meaning |
| --- | --- |
| MATCH | the text is there and the file value rounds to the printed value |
| MISMATCH | the text is there but the file value does not round to the printed value |
| TEXT_MISSING | the quoted text (or the number inside it) is no longer in the document |
| SOURCE_MISSING | the results file, the column or the row the selector names does not exist |
| UNTRACED | no results file for this number was found; the note says why |

**What "rounds to" means.** The printed number's last digit sets the precision. "5.8" has one decimal, so
any file value from 5.75 to 5.85 matches (5.84 matches, 5.86 does not, 5.9 does not). "-0.057" matches
-0.0574 but not -0.0576. "0.0006" matches 0.00063 but not 0.00066, which rounds to 0.0007. A value exactly
half-way counts as a match.

**Percent against fraction.** A file often stores 0.535 where the text says 54%. The check never guesses
this: a fraction is turned into a percent only where the registry row says `x100`. Other transforms the
registry can ask for are `neg` (the files store "-5.8% loss" where the text says "5.8% better"),
`ratio_pct` (a ratio of 1.21 printed as "+21%"), `expm1_pct` (a log difference printed as a percent),
`one_minus`, `half`, `mul=X` and `divby=X`. Several cells can be combined with `sum`, `diff`, `div`, `mul`,
`max` or `min`, for example "the largest of the three p-values is below 0.001". Other comparison modes are
`lt` (for "p < 0.001"), `oom` (same power of ten, for "as small as 10^-256"), `date` (so "14 November
2022" matches "2022-11-14"), `rel=X` (for "about 100") and `exact`.

**Three numbers are computed, not looked up.** Three numbers in the text are computed by the project's
code only in passing and are not stored in any file: the share of days on which the hybrid beats mcsGARCH
(71%, 74%, 54%), the share of mcsGARCH's squared error that comes from its ten worst forecasts (97%) and
the share of the hybrid's gain made at 09:15 (20% and 29%). The script recomputes them from the committed
forecast files (`selector` starting with `derived:`), using the same QLIKE formula as
`niftygex.evaluation.qlike`. The 78% of gross profit from five trades in F2_ECONOMICS.md is computed from
`results/f2_economics/trades.csv` the same way.

**Coverage scan.** The registry holds the headline numbers, not every digit. To show what it leaves out, the
script also lists every number in `PITCH_PANDEY.md` that no registry row covers
(`results/number_trace/uncovered_pitch.csv`). Dates, years, times of day, list numbers, "Table 4" style
references, identifiers such as R2 or SHA-256, links, code, formulas and the forecast horizons "15
minutes", "30 minutes" and "60 minutes" are skipped. The same scan runs over the working paper's main text
(up to its references), but because the paper is private that list is only printed to the screen or
written to a file outside the repository; it is never saved in the repository.

## How to run it

```
python scripts/trace_numbers.py
python scripts/trace_numbers.py --paper /path/to/WORKING_PAPER.md
python scripts/trace_numbers.py --paper /path/to/WORKING_PAPER.md --paper-uncovered-out /tmp/paper_uncovered.csv
OMP_NUM_THREADS=1 python -m pytest -q tests/test_trace_numbers.py
```

The first form checks the repository documents only (the paper's rows are still checked against the
results files, but their text is reported as "not checked"). The second also checks the paper's text and
prints the paper's uncovered numbers. The third saves that list instead; the script refuses a path inside
the repository. It takes about a second and writes:

* `results/number_trace/trace.csv`: one row per registry entry, with the printed value, the file value,
  the value after transforms, the status and the reason.
* `results/number_trace/uncovered_pitch.csv`: numbers in the pitch with no registry entry (line, number,
  context).

To add a number, add a row to `docs/number_registry.csv` and rerun. The registry rows for the paper quote
at most about ten words.

## Results (run of 3 November 2026, committed results at 9634054)

**221 registry entries: 210 MATCH, 7 MISMATCH, 0 TEXT_MISSING, 0 SOURCE_MISSING, 4 UNTRACED.**

| Document | Entries | MATCH | MISMATCH | UNTRACED |
| --- | --- | --- | --- | --- |
| PITCH_PANDEY.md | 69 | 66 | 2 | 1 |
| README.md | 9 | 7 | 0 | 2 |
| docs/BRIEFING_PANDEY.md | 10 | 10 | 0 | 0 |
| docs/REAL_DATA_RESULTS.md | 43 | 37 | 5 | 1 |
| docs/MODEL_ITERATIONS_REAL.md | 9 | 9 | 0 | 0 |
| docs/FALSIFICATION.md | 10 | 10 | 0 | 0 |
| docs/F2_ECONOMICS.md | 8 | 8 | 0 | 0 |
| docs/EVENT_STUDY.md | 7 | 7 | 0 | 0 |
| docs/PINNING_STUDY.md | 7 | 7 | 0 | 0 |
| docs/SENSEX_SWAP.md | 7 | 7 | 0 | 0 |
| docs/REPLICATION_NSE.md | 8 | 8 | 0 | 0 |
| docs/DEALER_SIGN.md | 6 | 6 | 0 | 0 |
| Working paper (private) | 28 | 28 | 0 | 0 |

Every pitch number that is a result now has a registry entry. The pitch scan leaves three numbers
uncovered, and none of them is a result: "360 minutes a day" (the sample design), "a 100 bp move" (the
scale of an illustration) and "a 50-point strike" (NSE's strike spacing). The working paper's main text has
many more numbers than the 28 registered here; its uncovered list was printed for the author and not saved.

### Every mismatch

| Claim | Where | Printed | File value | Source | My reading |
| --- | --- | --- | --- | --- | --- |
| PI36 | PITCH_PANDEY.md, point 4: "it beats 97% of them, 99.6% with controls" | 99.6% | 100% of the 288 usable random sign rules have a smaller \|t\| (one minus the placebo rank p 0.00346 gives 99.65%) | `results/falsification/t3_sign_rule_cloud_summary.csv`, row `t_x_neg_controls` | The file is right. 99.65 rounds to 99.7, not 99.6, and on the share reading used for the 97% it is 100%. Suggest "all 288 of them with controls" or "99.7%". |
| PI42 | PITCH_PANDEY.md, point 4: "it breaks even at about 0.5 bp of slippage per side" | 0.5 | 0.5546 | `results/f2_economics/metrics.csv`, rule P, test, base costs | The file is right; 0.5546 rounds to 0.6, and F2_ECONOMICS.md and REAL_DATA_RESULTS.md print 0.55. Suggest "about 0.55 bp". Minor, since the sentence says "about". |
| RD30 | REAL_DATA_RESULTS.md, R5 table, Pro row, regime-specific slopes | -0.069 | -0.0695 | `results/momentum_gex/regressions.csv` (929 days) | See the note under the table. The file named as the source gives -0.070. |
| RD31 | same cell, p-value | 0.0006 | 0.00066 | same | The file gives 0.0007. |
| RD33 | REAL_DATA_RESULTS.md, R5 table, US row, plain slope p-value | 0.59 | 0.584 | same | The file gives 0.58. |
| RD35 | REAL_DATA_RESULTS.md, R5 table, Client-mirror row, regime-specific slopes | -0.069 | -0.0683 | same | The file gives -0.068. |
| RD36 | REAL_DATA_RESULTS.md, R5 table, Client-mirror row, with controls, p-value | 0.95 | 0.941 | same | The file gives 0.94. |

**The five R5 table cells have one cause.** The section says the GEX regressions use 929 days and names
`results/momentum_gex/regressions.csv` as the source. Those five cells instead match the 932-day version of
the same regressions that the falsification study reran from the original script
(`results/falsification/t7_family_A_original_tests.csv`: -0.0694, p 0.00063, p 0.586, -0.0685, p 0.946,
which round to exactly the printed values). So the table mixes two runs. The file named as the source is
the one to follow, and updating the five cells to -0.070 (0.0007), 0.58, -0.068 and 0.94 changes no
conclusion: the Pro interaction stays clearly negative and the US and Client-mirror controls results stay
null.

No mismatch was found in the working paper's 28 registered numbers, in the briefing, or in any of the
study documents other than REAL_DATA_RESULTS.md.

### Untraced numbers

| Claim | Where | Printed | Why no source |
| --- | --- | --- | --- |
| PI68 | PITCH_PANDEY.md, "Two possible papers" table | 2 weeks | A time estimate in the plan, not a result. |
| RM08 | README.md, quick start | about 9 minutes on 4 cores | A run time. The first real-data run's `run_info.json` records 500 s (8.3 minutes) and the five-seed run behind the headline numbers records 5,962 s (99 minutes). No file records 9 minutes, so the README's figure is worth checking against the command it sits next to. |
| RM09 | README.md | 259 tests | The size of the test suite, not a results file. On this branch `OMP_NUM_THREADS=1 python -m pytest -q` passes 397 tests, 92 of them this check's own, so the suite had 305 before it; the README's 259 looks out of date. |
| RD43 | REAL_DATA_RESULTS.md, R1, MSE paragraph (and MODEL_ITERATIONS_REAL.md, finding 4) | p 0.13 to 0.18 | The lower end (0.134) matches the all-day MSE tests at 15 and 30 minutes, which run from 0.134 to 0.151. Only the hold-out HAR test at 15 minutes gives 0.180, so it is unclear which tests the upper end covers. Suggest "0.13 to 0.15" or naming the tests. |

### Other things noticed (not checked by the script)

* PINNING_STUDY.md calls the follow-up "a pre-registered test of five market-structure explanations",
  while PITCH_PANDEY.md and BRIEFING_PANDEY.md say "six". The registry checks numbers written as digits,
  not as words, so this is only flagged here.
* The R2 "+11% before November 2024" and the "about 100 weekly expiries since November 2024" (102 in the
  data) both check out; the latter is registered with a 5% tolerance because the text says "about".

## Limits of this check

* It checks that printed numbers agree with the committed result files. It does not check that those files
  are correct; that is what the tests and the reproduction run are for.
* The registry was written by hand by reading each results file. A wrong selector would show up as a
  MISMATCH or SOURCE_MISSING, not as a false MATCH, unless it happens to point at a different cell with the
  same rounded value; the notes say which row and column each entry uses so a reader can verify it.
* 221 entries cover the headline numbers. Secondary numbers (most table cells in the study documents, and
  most of the working paper) are not registered; the pitch scan shows that every remaining pitch number is
  a design setting.
* The check reads the worktree's committed results. If a full reproduction rewrites `results/`, rerun it.

## Files

* `scripts/trace_numbers.py`: the check.
* `docs/number_registry.csv`: the registry (221 rows).
* `tests/test_trace_numbers.py`: tests on small synthetic files (selectors, rounding, transforms, every
  status, the coverage scan and the rule that the paper's uncovered list never lands in the repository).
* `results/number_trace/trace.csv` and `results/number_trace/uncovered_pitch.csv`: the outputs.
