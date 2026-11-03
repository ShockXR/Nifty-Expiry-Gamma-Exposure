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
| `quoted_text` | a short piece of the sentence around it | `forecasts the next 15 and 30 minutes 5.8% and 7.6% better` |
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
2022" matches "2022-11-14"), `rel=X` (for "about 100") and `exact`. A quantity written in words ("all",
"four", "one in five") is read as an exact number (1, 4, 0.2), so where the text says "about one in five"
the row gives an explicit tolerance (`abs=0.05`).

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

## Results (run of 3 November 2026, after the pitch was shortened)

**281 registry entries: 270 MATCH, 5 MISMATCH, 0 TEXT_MISSING, 0 SOURCE_MISSING, 6 UNTRACED.**

| Document | Entries | MATCH | MISMATCH | UNTRACED |
| --- | --- | --- | --- | --- |
| PITCH_PANDEY.md | 69 | 64 | 0 | 5 |
| README.md | 12 | 11 | 0 | 1 |
| docs/BRIEFING_PANDEY.md | 22 | 22 | 0 | 0 |
| docs/REAL_DATA_RESULTS.md | 84 | 79 | 5 | 0 |
| docs/MODEL_ITERATIONS_REAL.md | 9 | 9 | 0 | 0 |
| docs/FALSIFICATION.md | 10 | 10 | 0 | 0 |
| docs/F2_ECONOMICS.md | 8 | 8 | 0 | 0 |
| docs/EVENT_STUDY.md | 7 | 7 | 0 | 0 |
| docs/PINNING_STUDY.md | 7 | 7 | 0 | 0 |
| docs/SENSEX_SWAP.md | 7 | 7 | 0 | 0 |
| docs/REPLICATION_NSE.md | 8 | 8 | 0 | 0 |
| docs/DEALER_SIGN.md | 10 | 10 | 0 | 0 |
| Working paper (private) | 28 | 28 | 0 | 0 |

**What changed since the first run (221 entries at 9634054).** The pitch, README, briefing,
REAL_DATA_RESULTS.md and DEALER_SIGN.md were rewritten after three referee responses, which left 20 entries
TEXT_MISSING. Ten were re-quoted to the new wording with the same source (PI13 to PI16, PI35, PI36, PI42,
BR03, BR04, DS03). Ten were removed because their numbers are no longer printed: PI17 to PI25 (the pitch's
expiry-day, after-the-open and hold-out margins and its 54% daily win share) and RM06 (the README's 54%);
their IDs are not reused. 72 entries were added for the new numbers, from `results/referee_forecast/` (the
4.8% and 5.9% margins over the best mcsGARCH given VIX or implied variance, the 1.7% (p 0.23) in the last
hour, the 31 challengers, Hansen's SPA and step-down p-values, plain LightGBM against the hybrid (p 0.32 to
0.97), the ridge 3.7% and 3.9%, and the EWMA fallback check: 5.5% and 6.9%, 42 days, 5.9%, 7.9% and
6.5%), from `results/referee_f2/` (the -0.029 unconditional slope, 95% and "about one in five" negative-Pro
evenings, 466 days, the largest robust p 0.012 and 0.038, 916 of 932 evenings, and the angle tests p 0.07
to 0.28 and 0.002 to 0.004), from `results/synthetic_control/` (+0.12, p 0.95, about 1.4 log points, 19
donors, +0.28 against +0.19) and from `results/sensex_swap/` for the P92 summary in REAL_DATA_RESULTS.md.
The pitch fixes for PI36 ("all of them with controls") and PI42 ("about 0.55 bp") now match. RM08 (about 8
minutes) and RD43 (p 0.18) are now traced, see below.

**Pitch shortened (later on 3 November).** The pitch was cut from about 2,150 to about 1,750 words. 17
entries whose numbers are no longer printed were removed (PI26 to PI30, PI32 to PI37, PI72, PI74, PI76 and
PI79 to PI81), ten were re-quoted to the new wording, and twelve were added: the Bank Nifty replication of
the forecasting hybrid (3.9%, 5.0%, 660 days, from `results/pipeline_nse_banknifty/p91/hypotheses.csv`), the
clean-baseline Tuesday premia (63% in the last hour, 26% over the day, from
`results/referee_expiry/r2_clean_baseline.csv`), the raw Bank Nifty Wednesday change (+0.90, from
`results/referee_expiry/two_day_raw_differences.csv`), the two printed after-change hour totals that lie
below our intervals (66.44 and 70.97, from `results/replication_nse/tables34_connectedness.csv`) and four
settings recorded as UNTRACED. The briefing was shortened the same way: BR05 and BR09 were removed, BR07,
BR08 and BR15 re-quoted, and BR20 to BR24 added (63%, 3.9%, 5.0%, 4.5 points, p = 0.0006).

Every pitch number that is a result has a registry entry, and the pitch scan now leaves nothing uncovered.
The four numbers that are not results are registered as UNTRACED: "360 minutes a day" (the sample design),
"a 100 bp move" (the scale of an illustration), "a 50-point strike" (NSE's strike spacing) and "5%" (the
significance level). The working paper's main text has many more
numbers than the 28 registered here; its uncovered list was printed for the author and not saved.

### Every mismatch

| Claim | Where | Printed | File value | Source | My reading |
| --- | --- | --- | --- | --- | --- |
| RD30 | REAL_DATA_RESULTS.md, R5 table, Pro row, regime-specific slopes | -0.069 | -0.0695 | `results/momentum_gex/regressions.csv` (929 days) | See the note under the table. The file named as the source gives -0.070. |
| RD31 | same cell, p-value | 0.0006 | 0.00066 | same | The file gives 0.0007. |
| RD33 | REAL_DATA_RESULTS.md, R5 table, US row, plain slope p-value | 0.59 | 0.584 | same | The file gives 0.58. |
| RD35 | REAL_DATA_RESULTS.md, R5 table, Client-mirror row, regime-specific slopes | -0.069 | -0.0683 | same | The file gives -0.068. |
| RD36 | REAL_DATA_RESULTS.md, R5 table, Client-mirror row, with controls, p-value | 0.95 | 0.941 | same | The file gives 0.94. |

**The five R5 table cells have one cause.** The section says the GEX regressions use 929 days and names
`results/momentum_gex/regressions.csv` as the source. Those five cells instead match the 932-day version of
the same regressions that the falsification study reran from the original script
(`results/falsification/t7_family_A_original_tests.csv`: -0.0694, p 0.00063, p 0.586, -0.0685, p 0.946,
which round to exactly the printed values). So the table mixes two runs. Updating the five cells to the
named file's -0.070 (0.0007), 0.58, -0.068 and 0.94 would change no conclusion: the Pro interaction stays
clearly negative and the US and Client-mirror controls results stay null. They are left as MISMATCH on
purpose for now: a full reproduction is regenerating these files, and the table will be fixed against the
new ones.

No mismatch was found in the working paper's 28 registered numbers, in the pitch, the README, the briefing,
or in any of the study documents other than REAL_DATA_RESULTS.md.

### Untraced numbers

| Claim | Where | Printed | Why no source |
| --- | --- | --- | --- |
| PI68 | PITCH_PANDEY.md, "Two possible papers" table | 2 weeks | A time estimate in the plan, not a result. |
| RM09 | README.md | 259 tests | The size of the test suite, not a results file. On this branch `OMP_NUM_THREADS=1 python -m pytest -q` collects 465 tests, 102 of them this check's own; the README's 259 looks out of date. |

Two numbers untraced in the first run now have a source. RM08, the README's "about 8 minutes on 4 cores"
for the real-data pipeline, matches the first run's `results/pipeline_nse/run_info.json` (500 s, 8.3
minutes), and the new "takes about 100" for the five-seed study matches `results/pipeline_nse_v2/run_info.json`
(5,962 s, 99 minutes). RD43, the upper end of "p 0.13 to 0.18" for the MSE tests, is the range over every
model's all-day MSE test against mcsGARCH at 15 and 30 minutes in `results/pipeline_nse_v2/tests_next15.csv`
and `tests_next30.csv` (0.131 to 0.182), leaving out `mcsgarch_r63`, which is the benchmark itself refitted
every 63 days (p 0.94 and 0.96); RD42 and RD43 now take the minimum and maximum over those rows.
MODEL_ITERATIONS_REAL.md prints the same range in two places; it is not registered there.

### Other things noticed (not checked by the script)

* The first run flagged that PINNING_STUDY.md said "five market-structure explanations" where the pitch and
  briefing said "six"; it now says "six (five tested by regression)", so the three agree.
* "Largest p 0.012" and "0.038" for F2 under persistence-robust inference (pitch, REAL_DATA_RESULTS.md) are
  the largest over the methods in `results/referee_f2/item3_f2_inference.csv` other than the within-block
  shuffles, which REFEREE_RESPONSE_F2.md reports only for comparison (from 2025 the within-block shuffle of
  the coefficient gives 0.055). The registry rows say so.
* "About one in five" negative-Pro evenings from 2025 is 21.5% in the file; it is checked as 0.2 with a
  tolerance of 0.05.
* The R2 "+11% before November 2024" and the "about 100 weekly expiries since November 2024" (102 in the
  data) both check out; the latter is registered with a 5% tolerance because the text says "about".

## Limits of this check

* It checks that printed numbers agree with the committed result files. It does not check that those files
  are correct; that is what the tests and the reproduction run are for.
* The registry was written by hand by reading each results file. A wrong selector would show up as a
  MISMATCH or SOURCE_MISSING, not as a false MATCH, unless it happens to point at a different cell with the
  same rounded value; the notes say which row and column each entry uses so a reader can verify it.
* 281 entries cover the headline numbers. Secondary numbers (most table cells in the study documents, and
  most of the working paper) are not registered; the pitch scan shows that every remaining pitch number is
  a design setting.
* The check reads the worktree's committed results. If a full reproduction rewrites `results/`, rerun it.

## Files

* `scripts/trace_numbers.py`: the check.
* `docs/number_registry.csv`: the registry (281 rows).
* `tests/test_trace_numbers.py`: 102 tests on small synthetic files (selectors, rounding, transforms,
  quantities in words, every status, the coverage scan and the rule that the paper's uncovered list never
  lands in the repository).
* `results/number_trace/trace.csv` and `results/number_trace/uncovered_pitch.csv`: the outputs.
