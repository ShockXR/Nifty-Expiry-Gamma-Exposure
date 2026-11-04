# Post-merge health check

Date: 4 November 2026, 01:56 to 02:20 UTC. Code state: main at merge commit 7c6f5f7 (pull requests #1, #2 and #3), with other agents' uncommitted work present in the same folder at the time. Every check below was run by an AI agent (Claude), not by a human. Labels: VERIFIED, PARTIALLY VERIFIED, FAILED, NOT RUN. Scratch outputs are in the agent's scratch folder, not in the repository.

## 1. Merge check

| Check | Result | Label |
| --- | --- | --- |
| `git diff origin/claude/project-thread-tddcw4 origin/main --stat` (PR #3 head against main) | empty: main has exactly PR #3's files | VERIFIED |
| PR #3 head 2b2bd22 is an ancestor of main | yes | VERIFIED |
| PR #2 head adbc553 (`claude/project-thread-rbrfe4`) is an ancestor of main | yes | VERIFIED |
| PR #1 head 22e2cbd (`claude/project-thread-vsf9l5`) is an ancestor of main | yes | VERIFIED |

Nothing from the three pull requests is missing on main. The pull requests were stacked (PR #3 contains PR #2, which contains PR #1), so the single merge brought in all three.

## 2. Tests

Full suite, run in the background with `OMP_NUM_THREADS=1` (Python 3.13, pinned versions), 01:56 to about 02:03 UTC.

| Result | Count |
| --- | --- |
| passed | 742 |
| failed | 1 (`tests/test_reproduce_all.py::test_every_registered_study_has_a_step`) |
| skipped | 7 (real Bank Nifty and Fin Nifty files absent when those tests ran) |
| expected failures (xfail) | 3 |

Label: PARTIALLY VERIFIED. The one failure is not a defect on main. It is caused by another agent's new, uncommitted plan `docs/prereg/A5_PERTURBATION.md`: the test requires every registered plan to have a step in `REGISTERED_RESULTS` or an entry in `PENDING` in `scripts/reproduce_all.py`, and A5 has neither yet. Rerun of that file at 02:12 UTC: same single failure, same reason. Fix before committing A5: add an A5 step or a `PENDING` entry.

Rerun at about 02:15 UTC of `test_reproduce_all.py`, `test_gex.py`, the new `test_gex_validation.py` (agent B), `test_audit_agent1.py` and the two real-data symbol files, after the raw download had finished: 109 passed, 1 failed (the A5 test above), 3 skipped (Fin Nifty raw files not downloaded), 3 xfailed. The 4 Bank Nifty real-data tests, skipped in the committed counts, now run and pass. No GEX test failed. The test count of 753 printed in README.md will change when the other agents' new test files are committed; it was left alone.

Note on reading the log: `pyproject.toml` sets `addopts = "-q"`, so `pytest -q` becomes very quiet and prints no final count line; counts above were taken from the progress characters. Use `-o addopts=""` to get the summary line.

## 3. Number trace

`python scripts/trace_numbers.py`: 678 registry entries, 664 MATCH, 0 MISMATCH, 0 TEXT_MISSING, 0 SOURCE_MISSING, 14 UNTRACED. Matches the expected figures. VERIFIED. (The script rewrites `results/number_trace/uncovered_pitch.csv`; its content did not change.)

## 4. Document fixes

Files checked: README.md, PITCH_PANDEY.md, 44 files in docs/, 10 in reports/, ITERATION_LOG.md (all 91 tracked Markdown files for dashes).

* (a) Dashes: 0 em-dashes and 0 en-dashes in any tracked Markdown file. VERIFIED, nothing to change.
* (b) Links and paths: a small checker read every Markdown link and every back-quoted file path. All links resolve. 23 back-quoted paths do not resolve from the repository root, but 22 are short forms given in a sentence that names their folder (for example `p104/hypotheses.csv` under `results/pipeline_nse_finnifty/`, `figures/lr_profiles.png` under `results/dealer_sign/`) or file-name patterns such as `ind_close_all_DDMMYYYY.csv`; all 22 exist. One is stale: ITERATION_LOG.md line 494 names `results/real_day_level/regressions.csv`, which no longer exists (the folder now has `r3_gex_regressions.csv` and others). It is a dated log entry, so it was left as written. PARTIALLY VERIFIED.
* (c) Stale statements: no `__P119__` placeholder anywhere; every P119 mention says it was not committed. Two stale statements fixed (table). Branch names in docs/P114 and docs/P117 describe where commits were pushed at the time and were left. Old test counts in study documents (620, 658, 668, 697, 705 and so on) are dated records and were left.
* (d) README commands, run in a clean copy of the commit (`git archive`) so nothing in results/ was touched: `pytest` (section 2); `scripts/demo_forecast_day.py --date 2025-12-23` ran in 0.7 s and wrote the page (VERIFIED); `experiments/paper_checks.py` ran in 27 s and its outputs are identical to the committed `results/paper_checks/` (VERIFIED); `experiments/inference_checks.py` did not finish within 200 s (NOT RUN to the end; over the 2-minute budget, not a failure). The synthetic pipeline (about 15 minutes), the power study and all real-data commands: NOT RUN as cheap commands.

| File | Edit (wording only, no result number changed) |
| --- | --- |
| reports/EXECUTION_SUMMARY.md | "How to resume": `git checkout claude/project-thread-tddcw4` replaced by `git checkout main`, with a comment that PRs #1 to #3 were merged into main (7c6f5f7) on 4 November 2026 |
| reports/MASTER_BACKTEST_REPORT.md | Line 3: added "(all three pull requests were merged into main on 4 November 2026, merge commit 7c6f5f7)" after "the validation changes on this branch" |

## 5. CI

`.github/workflows/tests.yml` runs `pytest -q` on Python 3.11, 3.12 and 3.13 on every push to main and every pull request, after `pip install ".[download]" pytest`. Latest run on main: run 149 on 7c6f5f7 (the merge), success on all three versions (3.11 took 3 min 20 s of tests, 3.12 5 min 20 s, 3.13 3 min 39 s). The PR #3 run (148) and the last PR #2 runs (142 to 147) also succeeded. VERIFIED. Two small points: CI installs the latest versions of the dependencies, not `requirements-tested.txt`, so a new pandas or numpy release could break CI without a code change; and GitHub warns that `actions/checkout@v4` and `actions/setup-python@v5` target the deprecated Node.js 20 (they are forced onto Node.js 24 for now). Bumping them to `actions/checkout@v5` and `actions/setup-python@v6`, which run on Node.js 24, would clear the warning; this was not done here.

Added after this check: a fourth CI job, `pytest-pinned` (Python 3.13), installs `requirements-tested.txt` and then the package with `--no-deps`, so the pinned versions that produced the committed numbers are tested too. The three unpinned jobs are unchanged. The workflow file still parses as YAML. The pinned versions are the ones installed on this machine under Python 3.13.18. The new job has not run on GitHub yet. README Quick start now adds `pip install -e ".[download]"` before the real-data commands, because the download scripts need `requests`.

## 6. Raw-data integrity

The background download (`scripts/download_nse.py --start 2023-01-01 --end 2026-11-03`) ended with EXIT 0 at about 02:11 UTC; the minute-bar fetch had already ended with EXIT 0. It downloads bhavcopies, participant OI, index closes, `index_daily.csv` and `india_vix.csv` only.

(a) `data/MANIFEST.csv` (5,213 files), hash by hash:

| Group | Match | Different | Missing |
| --- | --- | --- | --- |
| bhavcopy | 944 | 0 | 0 |
| participant_oi | 944 | 0 | 0 |
| index_close | 944 | 0 | 0 |
| banknifty_1min.csv | 1 | 0 | 0 |
| nifty_1min.csv, index_daily.csv, india_vix.csv | 0 | 3 | 0 |
| minute_bars_source.json | 0 | 1 | 0 |
| nse_full, finnifty, extra_2022, finnifty and sensex minute files | 0 | 0 | 2,376 |
| Total | 2,833 | 4 | 2,376 |

Extra on disk: the three 3 November 2026 daily files, as expected. The 4 differences are all explained by the new data running one day longer: after dropping the rows dated 3 November 2026, `nifty_1min.csv`, `index_daily.csv` and `india_vix.csv` are byte for byte identical to the manifest (same size and SHA-256). `minute_bars_source.json` differs because the fetch was run without `--nifty-ref` and `--until`, so it records the provider's newest commit (7a80e0d3) and a last bar on 3 November instead of the pinned commit (7f94ced8). The 2,376 missing files come from other download scripts (`fetch_finnifty_data.py`, the full-bhavcopy and Sensex downloads) that were not run. Label: VERIFIED for the 2,837 files downloaded (every one matches the committed run up to 2 November 2026); NOT RUN for the 2,376 others.

(b) `scripts/validate_minute_bars.py --raw data/raw --start 2023-01-01 --out <scratch>`: on every day already in the committed `minute_vs_official_*.csv` files (949 Nifty rows, 817 Bank Nifty rows) every column is identical. The only change is the extra day 3 November 2026 for Nifty, where open, high, low and close match NSE exactly. Summary: Nifty highs within 5 bp on 944 of 944 days, lows on 943 of 944 (committed: 943 of 943 and 942 of 943); Bank Nifty unchanged (811 days compared). VERIFIED.

(c) Five random bhavcopy days (seed 20261104): 31 Jan 2023, 15 Nov 2023, 9 May 2024 (old layout), 22 Jul 2024, 28 Jan 2025 (new UDiFF layout). VERIFIED on all five:
* OI is in shares: the greatest common divisor of all open-interest values is 25 for Nifty and 5, 15 or 25 for Bank Nifty, a lot size or a common factor of two live lot sizes; contract counts would give 1. In the new layout 91% to 100% of rows are an exact multiple of their own lot size; the rest are rows whose lot changed (two lot sizes live, 25 and 75 for Nifty on 28 Jan 2025), as DATA_ACQUISITION.md fact 2 says.
* Sorting is not assumed: in the new layout 56 of 56 and 48 of 48 option groups are not in strike order in the raw file; the old layout was in order. The loader's output is in strike order on all five days.
* Expiry flag from traded contracts: contracts expiring that day traded on 15 Nov 2023 (Bank Nifty, a Wednesday) and 9 May 2024 (Nifty, a Thursday), and on none of the other three days, which matches the expiry calendar of that time.

## 7. Open items

1. Add an A5 entry (step or `PENDING`) to `scripts/reproduce_all.py` before committing `docs/prereg/A5_PERTURBATION.md`, or the suite fails.
2. README test count (753) needs updating once agents' new tests are committed.
3. (README install line fixed, see section 5.) `validate_minute_bars.py` with its default `--out` overwrites `results/data_validation/` (with a 3 November day unless the bars are fetched with `--until 2026-11-02`).
4. Watch the first run of the new pinned CI job; bump the actions to Node 24 versions.
5. NOT RUN: Fin Nifty, full bhavcopy, 2022 and Sensex raw files; `inference_checks.py`, the synthetic pipeline, the power study and all real-data pipeline commands.
