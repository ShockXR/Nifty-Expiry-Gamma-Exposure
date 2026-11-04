# Master validation report: Nifty expiry gamma exposure and volatility forecasting (niftygex)

Date: 3 November 2026. Code state: PR #2 head (adbc553) plus the validation changes on this branch (all three pull requests were merged into main on 4 November 2026, merge commit 7c6f5f7). All checks were run by AI agents (Sonnet), not by a human reviewer, and every claim below says where its evidence is. Nothing here is a trading result: the project forecasts intraday volatility, so it is judged on forecast accuracy, statistical tests, replication of a published paper and honest accounting of how many tests were run.

## 1. Summary in plain language

1. **The headline forecast result reproduces.** Starting only from the committed forecast files, a separate piece of code written for this audit (it does not import `niftygex`) gives the same loss for all 32 models and the same gains: the hybrid model is 5.80% and 7.65% better than our version of mcsGARCH at 15 and 30 minutes, and 4.43% and 6.64% better than mcsGARCH as the paper specifies it. The Diebold-Mariano test statistics (6.409, 8.340, 3.801) match the committed ones. [VERIFIED, REAL data, OUR CHECK]
2. **The 15 and 30 minute gain is stable.** It is positive in each half of the test period, in each calendar year, in all 11 rolling half-year windows, after dropping the most extreme observations, and after dropping the closing-auction days. [VERIFIED]
3. **The last-hour gain is weak.** The hybrid wins on only 53.5% of days (sign test p = 0.0506), shows no gain in 2025 (-0.01%) and loses in 2 of 11 rolling windows. Say "not supported", not "6%". [PARTIALLY VERIFIED]
4. **One real weakness in the code is not fixed.** The choice between the daily EGARCH model and its simpler EWMA fallback looks at days after the forecast date. The authors had already disclosed and sized it (2 of 38 refits, 42 test days; hybrid margins 5.9%, 7.9% and 6.5% without those days). We reproduced the effect on simulated data and found that the audit's own probe can move early forecasts by up to 2.4 times when the future is replaced. Its effect on the committed headline was NOT measured, because the raw data are not in the repository. [FAILED probe, impact UNVERIFIED]
5. **The option maths is right; input checking is weak.** Put-call parity, gamma, vega, vanna, charm, implied-volatility round trips and the GEX formula all match independent checks (32 of 32 numeric checks). But a failed or expired option chain silently gives net GEX of exactly 0.0, and the strings CE and PE are all read as calls. 24 of 41 stress cases were silently accepted. [VERIFIED, defects open]

**Presentation readiness: ready for a professor meeting WITH the listed limitations, not ready for production.** Gate checks are in section 7.

## 2. What was examined

| Item | Detail |
|---|---|
| Prototype | `niftygex`: GEX engine (Black-76, five sign rules coded, three used), realized measures, mcsGARCH, forecast ladder (naive, HAR, VIX, mcsGARCH, LightGBM, hybrid), walk-forward evaluation, replication of the 2025 expiry paper, 37 registered studies |
| Data | REAL: NSE bhavcopy, participant OI, India VIX, third-party 1-minute bars for Nifty, Bank Nifty and Fin Nifty (raw files are not in the repository, hashes are in `data/MANIFEST.csv`). SYNTHETIC: `niftygex.simulate`. Test set 792 days, 10 Aug 2023 to 2 Nov 2026 |
| Environment | Python 3.13, 4 cores, pinned versions in `requirements-tested.txt`; CI also runs 3.11 and 3.12 |
| Baselines compared | naive, HAR, VIX, mcsGARCH (our version and as printed in the paper), plain LightGBM |
| Not done | Rerun of the pipeline from raw data (about 25 minutes of downloads and about 2 hours of compute), so the effect of the open look-ahead and all perturbation runs (A5) are BLOCKED |

## 3. Baseline against revised results

| Measure | Before this validation | After | Explained by |
|---|---|---|---|
| Tests | 708 passed, 7 skipped (271 s) | 743 passed, 7 skipped, 3 expected failures that pin known defects | 38 new audit tests |
| Printed numbers traced to result files | 664 match, 14 untraced, 0 mismatch | same | no number was changed |
| Headline forecast gains | 5.8/7.6% and 4.4/6.6% | unchanged | recomputed independently |
| Figures | untagged scoreboards | 62 figures redrawn in place with REAL or SYNTHETIC tag, day count and development or hold-out note | same CSV numbers, labels only |
| Code changes | none | Diebold-Mariano test no longer crashes on empty input; docstrings corrected (Rogers-Satchell is biased low under discrete sampling: 0.79, 0.93, 0.99 at 60, 600, 6,000 sub-steps); plot labels; `run_pipeline.py` passes the data label | see `CODE_AUDIT.md` |
| Documents | stale test counts, "same information" wording, an uncaveated +1.14 event-study headline | corrected | see `METHODOLOGY_VALIDATION.md` |

## 4. Evidence by dimension

| Dimension | Finding | State |
|---|---|---|
| Functional correctness | 20 equations or definitions checked against code, 32 of 32 numeric checks pass; walk-forward features are causal; forecasts are causal except the EGARCH gate | VERIFIED, with one FAILED probe |
| Experimental performance | see section 1, points 1 to 3 | VERIFIED / PARTIALLY VERIFIED |
| Is it information or just a learnt correction? | Plain LightGBM matches the hybrid; the placebo with shuffled training labels removes the gain (+4.9% to -19%); a day-permutation control still leaves some gain, so beating mcsGARCH is not by itself proof of option-market information. This agrees with the repo's own studies P106 and P110 | PARTIALLY VERIFIED |
| Reproducibility | Tests, number tracer and demo pages reproduce; full pipeline rerun not repeated here | PARTIALLY VERIFIED |
| Robustness | GEX engine: 41 failure cases (24 silent, see `RISK_REGISTER.md`) | VERIFIED (cases run), defects open |
| Presentation | Demo flow tested (`DEMO_WALKTHROUGH.md`); six deck slides overflow a 1080 pixel stage and need a check in the real viewer | PARTIALLY VERIFIED |

## 5. Data checks

The 1-minute bars match the hashes in the manifest and show no duplicates or timestamp problems in the files checked; a 69-day sample agrees with NSE highs and lows (the full 943-day comparison was not rerun here, the authors' own run reports 943 of 943 highs within 0.05%). Not fetched here: bhavcopy and participant files. [PARTIALLY VERIFIED]

## 6. Open items, most serious first

1. EGARCH versus EWMA gate and the full-sample EWMA floor (A4-A-001, A1-A-002). Fix: filter and choose using data up to each block end only, then rerun the pipeline and report the change in the headline. Until then quote the authors' own sensitivity (margins without the 42 affected days).
2. GEX engine accepts bad input silently (A4-A-002, A4-A-003). Not fixed because a fix could change GEX panels; document as a limitation, add input validation in the next round. **Fixed on 4 November 2026** with input checks that change no committed GEX panel (`EXECUTION_SUMMARY.md`, follow-up).
3. Last-hour result is not supported; keep it out of the pitch headline.
4. A5 perturbation rerun and the full raw-data comparison are BLOCKED.
5. `docs/TEST_LEDGER.md` does not regenerate exactly in a fresh checkout because one column depends on the original clone's git history (A1-A-013). Do not regenerate it.
6. Deck: six slides need a visual check in the real viewer (A3-A-005); five diurnal-curve figures are dominated by the 09:15 spike and need the raw data to redraw (A3-A-003).

## 7. Presentation-readiness gates

| Gate | State |
|---|---|
| Primary functionality demonstrated | VERIFIED (demo run from committed files, byte-identical to a fresh run) |
| Critical calculations checked | VERIFIED |
| Important tests executed | VERIFIED (743 pass) |
| Major risks documented | VERIFIED (`RISK_REGISTER.md`) |
| Charts correspond to results | PARTIALLY VERIFIED (a sample was viewed, numbers spot-checked) |
| Documentation matches implementation | VERIFIED for traced numbers, PARTIALLY for prose |
| Demo path validated | VERIFIED |
| Performance claims defensible | VERIFIED for 15 and 30 minutes, not for the last hour |

Reports in this folder: `CODE_AUDIT.md`, `REPRODUCIBILITY.md`, `METHODOLOGY_VALIDATION.md`, `PROFESSOR_DEFENSE.md`, `VISUAL_QA.md`, `DEMO_WALKTHROUGH.md`, `BACKTEST_STRESS_REPORT.md`, `RISK_REGISTER.md`, `EXECUTION_SUMMARY.md`, and `findings/` (49 findings in CSV).
