# Independent review of the real-data studies

*Note added 3 November 2026: numbers quoted in this review are from the result files as they stood when it was written. Where a later fix or rerun changed a number, `docs/REAL_DATA_RESULTS.md` and the study documents have the current value; how each point was addressed is listed there and in `ITERATION_LOG.md` (iteration 20). The first-run files this review cites from `results/closing_auction/` (`r1_dm_tests.csv`, `r1_scoreboards.csv`) and `results/forecast_diagnostics/` were rewritten from the second run by the reproduction of 3 November 2026; the versions it checked are kept in `results/archive_v1_first_run/`.*

*Note added 3 November 2026 after study P116:* here "mcsGARCH" and "the paper's model" mean our implementation (a separate time-of-day curve for expiry days, a 5-minute smoothing of the curves, a zero mean), not the paper's own specification. Against the paper's specification (one curve, no smoothing, ARMA(1,1) mean) see `docs/P116_BENCHMARK_AS_PRINTED.md`; there the hybrid's Nifty gain is 4.43% and 6.64% at 15 and 30 minutes.

Reviewer: a separate AI agent (Claude), instructed to act as an outside, skeptical referee; not a human reviewer. Date: 2 November 2026.
Scope: `docs/REAL_DATA_RESULTS.md` (R0 to R5, the version corrected for the closing auction in commit b2bfcc9), `docs/REPLICATION_NSE.md`, `docs/PINNING_STUDY.md`, the code in `src/niftygex/` and `experiments/`, the result files under `results/`, and the raw data under `data/raw/` (read only). The paper itself was not opened, as instructed, so every "printed" value from the paper is taken as typed in `experiments/replicate_paper.py`.

What I did: read every file listed above; re-ran `experiments/real_day_level.py` and `experiments/momentum_gex.py` into a scratch folder (both reproduce the committed result files byte for byte); did not re-run `scripts/run_pipeline.py` (about nine minutes) or the replication and pinning scripts, so for those I checked the documents against the committed result files and the code paths that produced them; wrote small scripts that recompute summary statistics from `results/pipeline_nse/forecasts_*.csv` and the feature panel. One slip to disclose: one compound shell command ended with a read-only `git status`; nothing was changed or committed, and no other git command was run.

Terms used below, defined once. **Realized variance**: the sum of squared one-minute returns over a window. **QLIKE**: a loss for variance forecasts, y/f minus log(y/f) minus 1 where y is realized and f is forecast; it punishes forecasts that are too low more than forecasts that are too high. **MSE**: mean squared error, the plain average of (forecast minus realized) squared. **Diebold-Mariano (DM) test**: a t-type test of whether two forecasting models have the same average loss, here run on daily average losses with Newey-West standard errors (which allow the daily losses to be correlated over time). **Model Confidence Set (MCS)**: a procedure that keeps every model that cannot be shown to be worse than the best one at a chosen confidence level, so it handles the "many models" problem that single DM tests do not. **Walk-forward**: refitting a model only on data before the period being forecast. **Look-ahead**: any use of information that was not available at forecast time. **GEX**: gamma exposure, an estimate in rupees of how much option dealers must buy or sell per 1% index move; its sign depends on who is assumed to be the dealer, which the data do not say. **mcsGARCH**: the paper's model, a daily GARCH times a time-of-day pattern times an intraday GARCH. **HAR**: a regression of future variance on past variance at several horizons. **LightGBM**: a gradient-boosted tree model. **Permutation test**: a test that reshuffles labels many times to see how often chance alone produces the observed gap. **Holm correction**: a rule for adjusting p-values when several tests are run together. **bp**: basis point, one hundredth of a percent.

---

## Verdict

This is unusually careful student work, and the documents are honest about their nulls. I checked 64 numbers across the three documents against the result files and every one of them is transcribed correctly (table at the end); the corrected closing-auction section is right, and the two experiments I could re-run reproduce exactly. I found no look-ahead in the forecasting ladder: the mcsGARCH filter is causal, the LightGBM models are trained only on earlier blocks, the GEX features come from the previous evening's chain, and the rolling scales are backward-looking. Units (GEX in rupees per 1% move, open interest in shares, Black-76 gamma, time in trading days) check out.

No issue I found overturns a headline claim. But four things would make a referee send this back: (1) the pipeline computes MSE as well as QLIKE and the document never mentions MSE, under which the model ranking reverses (mcsGARCH and the hybrids are the worst non-naive models); (2) much of the hybrid's QLIKE gain comes from the single 09:15 anchor, which the document does not say; (3) the method is misdescribed in one place (mcsGARCH is refitted every 63 days, not 21); and (4) the headline wording for finding 2 overstates the expiry-day gain. The remaining points are about scale comparability of GEX across sign rules, the number of tests run, and some replication caveats that the prose should state rather than leave in a table. Recommendation: accept the results as reported, after the edits below; the conclusions "GEX does not help forecasting", "no momentum, mild reversal", and "pinning is regime-specific" are well supported.

---

## Issues

### Critical

None. I looked for look-ahead, wrong units, mis-transcribed numbers, and claims the files do not support; after the closing-auction correction, every headline in the three documents is supported by the file it cites. This judgment rests on the committed `results/pipeline_nse/` files, which I did not regenerate.

### Major

**M1. MSE results exist but are not reported, and they reverse the ranking.**
Location: `results/pipeline_nse/scoreboard_next15.csv` (column `mse`), `scoreboard_next30.csv`; `docs/REAL_DATA_RESULTS.md` R1, lines 59 to 88, which mention only QLIKE.
Evidence (next 15 minutes, all days): MSE is 2.33e-10 for mcsGARCH, 1.48e-10 for the hybrid, but 3.5e-11 for LightGBM and HAR and 3.6e-11 for the VIX-only model. So under MSE the paper's model and the hybrids are six times worse than a plain HAR, while under QLIKE they are the best. The cause is a handful of opening anchors: the ten largest squared errors are 96.7% of mcsGARCH's total MSE, and the largest is 5 June 2024 at 09:15 (the day after the election-result crash), where mcsGARCH forecast a 15-minute variance of 0.00201 against a realized 0.000099, twenty times too high; the hybrid inherited most of it (0.00157), LightGBM did not (0.000008). Excluding the 09:15 anchor, mcsGARCH's MSE is still 1.5e-11 against 9.7e-12 for LightGBM and 9.4e-12 for HAR.
Why it matters: QLIKE is the right loss for variance forecasts and the document says so, but a reader who sees only QLIKE does not learn that the paper's model occasionally produces a wildly wrong forecast after a crash day, which is exactly what a risk manager would want to know.
Fix: add the `mse` column to the R1 table (or a second table), say in one sentence why the two losses disagree, and add a robustness line with the 09:15 anchor removed.

**M2. The hybrid's gain is concentrated at the opening anchor, and the pure LightGBM gain is not stable across regimes.**
Location: `results/pipeline_nse/forecasts_next15.csv`, `forecasts_next30.csv`; `docs/REAL_DATA_RESULTS.md` finding 2 and R1 table.
Evidence (QLIKE, % against mcsGARCH): at the 09:15 anchor the hybrid is 16.4% better (next 15) and 26.9% better (next 30), LightGBM 18.4% and 25.7% better, HAR 15.1% and 5.1% worse. Excluding that one anchor per day, the hybrid is 5.8% and 7.1% better (the headline says 6.7% and 9.0%), LightGBM only 2.5% and 1.9% better, HAR 4.3% and 6.9% worse. By expiry regime, LightGBM without GEX is worse than mcsGARCH in the first regime (+2.8% at 15 minutes, +4.8% at 30) and better in the later two (-6.2% and -8.5%; -8.3% and -10.8%); the hybrid is better in all three (-5.4%, -7.0%, -7.7% at 15 minutes).
Why it matters: the result "the paper's model can be beaten" is true, but a good part of the beating is at the open, where mcsGARCH relies on its time-of-day factor and yesterday's daily variance, and the claim that LightGBM alone beats mcsGARCH (finding 2 says "about 4%") is not true in the first 20 months.
Fix: add a table of QLIKE gain by anchor time (or a figure) and by regime; rewrite finding 2 to say where the gain comes from.

**M3. The walk-forward schedule is misdescribed.**
Location: `docs/REAL_DATA_RESULTS.md` line 61: "Every model is refitted every 21 trading days". Code: `src/niftygex/forecast.py` lines 45 to 46 (`block: int = 21`, `mcs_refit_every: int = 63`) and lines 119 to 121 (the mcsGARCH refit runs only when `start - last_refit >= cfg.mcs_refit_every`).
Evidence: mcsGARCH and the daily EGARCH that feeds it are refitted every 63 trading days, which is also what line 86 implies ("13 walk-forward refits" over 792 test days). LightGBM and HAR are refitted every 21 days.
Why it matters: a reader replicating from the document would get a different protocol; and the slower refit mildly handicaps the paper's model in the contest (it uses parameters up to 63 days old, the challengers up to 21).
Fix: state both schedules; ideally rerun with `mcs_refit_every=21` once to show the contest result does not depend on this (the fit is cheap relative to LightGBM).

**M4. Finding 2 overstates the expiry-day gain.**
Location: `docs/REAL_DATA_RESULTS.md` line 28: "about 7% and 9% better (QLIKE loss), on all days and on expiry days".
Evidence: `scoreboard_next15.csv` and `scoreboard_next30.csv`, expiry rows: hybrid (no GEX) is 6.2% and 6.6% better on expiry days; without the closing-auction days 5.5% and 5.7% (`results/closing_auction/r1_dm_tests.csv`). The table in R1 is right; the sentence is not.
Fix: "about 7% and 9% on all days, and about 6% on expiry days".

**M5. The GEX variable has a different scale under each sign rule and is mechanically larger on expiry eves, so the R3 coefficients are not comparable across rules.**
Location: `docs/REAL_DATA_RESULTS.md` line 108 ("so it is a share between about -0.3 and +0.6"), R3 table lines 110 to 114, finding 5; `src/niftygex/features.py` line 120 (20-day rolling median of gross GEX as the scale).
Evidence (from the feature panel): the ratio runs from -0.31 to +0.62 under the US rule, but only from -0.11 to +0.10 under the Pro rule and from -0.09 to +0.19 under the Client mirror. The "-0.3 to +0.6" sentence is true for one rule only. Because the Pro ratio is about five times smaller in spread, its coefficient (-2.07) is not "five times the US effect" (-0.38); per standard deviation the two are of similar size. Separately, the mean absolute Pro ratio is 0.029 when the chain has one day to expiry and 0.008 at five days, because gross gamma rises as expiry nears (gross GEX 1.0e12 rupees at one day against 2.9e11 at five) and a 20-day median does not remove that within-week swing. So the GEX regressor is itself about four times larger on expiry eves, which feeds the GEX x expiry interaction.
Why it matters: finding 5 ("a sign that flips on expiry day") is a headline, and part of what the interaction captures is a scale artefact of the construction, not necessarily a change in behaviour. R5 is not affected because it uses only the sign of GEX.
Fix: report coefficients per one standard deviation within each rule; re-estimate R3 with the sign of GEX (a dummy) instead of its level, which is scale-free; or scale by gross GEX of the same tenor. State the ratio range per rule.

**M6. The number of tests behind each table is larger than the text says, and only two tables adjust for it.**
Location: all three documents. Counts from the result files: R1 runs 84 one-sided DM tests (`tests_*.csv`, 28 per target) and adjusts only via the MCS; R2 has 18 rows (`r2_expiry_effect_by_regime.csv`: three measures times three regimes, plus three measures times three pairwise differences) while line 102 says "nine tests"; R3 has 12 GEX coefficients (three rules, two terms, with and without regime controls); R5 reports 9 interaction cells but `momentum_gex/regressions.csv` holds 27 sign-interaction tests (three rules, three specifications, three day subsets) plus 9 continuous ones; the pinning files hold 211 p-values and Holm is applied only across the three sign rules in `t5_gamma_conditioning.csv`.
Why it matters: with 27 interaction tests, one p of 0.0002 is still convincing, but the "Client mirror" cell at p = 0.00003 that disappears with controls, and the R2 Thursday-regime cell at p = 0.045, are the kind of result that multiplicity erodes.
Fix: for each table, say how many tests are in the family and give a Holm or Benjamini-Hochberg column, or at least mark which results survive a 5% family-wise threshold. For R1, lead with the MCS, which already does this, and add the direct comparison readers want: hybrid against HAR for the last hour (my run of the committed daily losses gives DM p = 0.0095 on all days in HAR's favour, including the auction days, and 0.032 on expiry days).

**M7. Replication: two before-window statistics do not match the printed ones and the weekday counts differ, but the prose does not say so; and the printed values themselves were typed by hand.**
Location: `docs/REPLICATION_NSE.md` Section 9 table (line 103) and findings 1, 8; `experiments/replicate_paper.py` (`P_T1` to `P_T6`).
Evidence: the inferred before window reproduces the Nifty maximum to nine digits (0.006221108412) and the Bank Nifty maximum (0.012023448), the N, the SD to three digits and the kurtosis to within one unit, but the minima differ: Nifty -0.008534 against printed -0.008585, Bank Nifty -0.017135 against printed -0.017257. Both minima are the 09:15 bar of 30 January 2023, the bar that carries the overnight gap, and both are 0.5 to 1.2 bp less negative than printed. That is what one would get if the paper computed the overnight gap from NSE's official close (a 30-minute average before August 2026) rather than from the last bar's close. The inferred window also has 13 Wednesday and 12 Thursday expiry days against the printed 13 and 13 (after window: 11 and 12 against 12 and 13). None of this is in the prose. Finally, the "printed" values in `replicate_paper.py` are hand-typed and, by the rules of this review, I could not check them against the PDF, so the "typesetting slips" in Section 8 could equally be transcription slips.
Fix: say in Section 9 that the maxima match exactly and the minima differ by 0.5 and 1.2 bp at the overnight bar; test the official-close gap variant (one line in `replication.py`); add the weekday-count mismatch to the list of things the inferred window does not reproduce; have a second person re-key every printed value with page and table references.

### Minor

**m1. Stale sentence.** `docs/REAL_DATA_RESULTS.md` line 150: "The 15:29 bar close is the last trade, not NSE's official close, which is a 30-minute average." Line 148 and `results/closing_auction/auction_evidence.csv` (median last-bar against official close 0.0 bp after 3 August 2026) say the opposite for the auction period. Fix: "Before 3 August 2026, the 15:29 bar close ...".

**m2. Bolding.** R1 table (line 66) bolds HAR's last-hour cells (-10.8, -10.2) while finding 3 and line 80 now say that edge is not established. Fix: remove the bold or bold the cells that survive (the hybrid at 15 and 30 minutes).

**m3. "Nine tests".** Line 102 says three regimes times three measures is nine tests; the file has 18 rows. Fix: "nine within-regime tests and nine pairwise comparisons".

**m4. One full-sample constant in a feature.** `src/niftygex/features.py` line 114: `sig_dc = np.sqrt(np.median(r2))` is the median squared return over the whole sample and feeds `bvc` and so `ofi15`. It is a single scale constant, so the look-ahead is immaterial, but it is look-ahead. Fix: compute it from the training window at each refit, or say in the caveats that it is a fixed constant.

**m5. DM p-values use the normal after the small-sample correction.** `src/niftygex/evaluation.py` lines 60 to 64 use `norm.cdf`; Harvey, Leybourne and Newbold pair their correction with a Student t with n-1 degrees of freedom. With n = 792 days it does not matter; with n = 169 expiry days it moves p in the third decimal. Fix: use `t.cdf(stat, n-1)`.

**m6. Version pins disagree with the run record.** `requirements-tested.txt` pins numpy 2.5.4 and pandas 3.1.0; `results/pipeline_nse/run_info.json` records numpy 2.5.3 and pandas 3.0.5. Fix: pin what was run, or rerun under the pins and update `run_info.json`.

**m7. Regime label as a LightGBM feature.** The regime dummy is a feature, so in the first 21-day block after each regime change the model has never seen that label. Not leakage, but it means the model cannot use the feature exactly when it would matter. Fix: note it, or replace the label with the expiry weekday, which is known in advance.

**m8. Attribution.** Line 130 credits Gao, Han, Li and Zhou (2018) with intraday momentum "in US index futures" from "the return over most of the day". That paper used the SPY ETF and the first half-hour return as the predictor; the rest-of-day design in futures is Baltussen, Da, Lammers and Martens (2021). Fix: reword.

**m9. QLIKE is tail-heavy here.** The top 1% of anchors carry 27 to 34% of total QLIKE loss for every model (largest single-anchor loss 188 at 15 minutes, 116 at 30 minutes, 35 for the last hour). The DM test on daily averages absorbs some of this. Fix: add a trimmed-mean or median QLIKE line as a robustness check.

**m10. Economic meaning of the gain is not given.** A 7% lower QLIKE has no obvious translation for a reader. Fix: one sentence, for example the implied change in a value-at-risk or option-hedging error, or the share of days on which the hybrid is closer.

**m11. Interest rate fixed at 6.5%** (`run_info.json`, `"rate": 0.065`) for all of 2023 to 2026. Gamma is insensitive to this, and the forward is backed out by parity, so no effect; say so in one line.

**m12. Dealer proxy.** The participant file also has an FII category, a natural third guess for the dealer side; only Pro and the Client mirror are used. Fix: list it as a planned fourth rule, or explain why not.

**m13. One special session handled differently in two studies.** The 21 October 2025 Muhurat session (60 bars) is dropped from the forecasting pipeline by the coverage rule (`src/niftygex/data.py` line 303) but is a day in the pinning panel (939 days in `results/pinning/data_checks.csv`). It is not an expiry day, so the tests are unaffected. Fix: one line in each document.

**m14. Pre-specification is asserted, not evidenced.** `docs/PINNING_STUDY.md` line 99 says P1 and P2 were "fixed in the script's docstring before any result was seen". Fix: cite the commit hash and time at which the docstring was committed.

**m15. Pipeline reproducibility is claimed but I did not test it.** Both experiment scripts I ran reproduce byte for byte, seeds are fixed everywhere, and `docs/REPRODUCIBILITY.md` is clear. The nine-minute pipeline should be rerun once by someone other than the author to confirm the scoreboard is identical (LightGBM with a fixed seed is deterministic on one machine but not necessarily across library versions, see m6).

---

## Questions a professor would ask

1. If I hand you a day at 09:14, which model should I use and how much worse is the paper's model, in rupees of hedging error? (M1, M2, m10)
2. Why does mcsGARCH forecast twenty times too much variance on 5 June 2024? Is it the daily EGARCH reacting to the 4 June crash, and would a Student-t daily model or a cap on the daily factor fix it? (M1)
3. Does the hybrid still win if mcsGARCH is refitted every 21 days like the other models? (M3)
4. In R3, is the expiry-day sign flip still there when GEX enters as a sign only, so that the expiry-eve rise in gross gamma cannot drive it? (M5)
5. In R5, why would a market with a reversal into the close show a stronger reversal when a proxy says dealers are short gamma, the opposite of the hedging story? The document says it is puzzling; what is the leading alternative, and is it testable (for example, Pro positioning as a proxy for retail option buying that gets unwound before 15:30)?
6. How many of the 27 interaction tests in R5 and the 12 GEX terms in R3 survive a Holm correction? (M6)
7. For the replication, if you recompute the overnight gap from NSE's official close, do the minima match the printed values? If so, you have learned how the paper built its returns. (M7)
8. The before window was found by searching 348 windows for the best match to Table 1. Which of the later tables do you treat as out-of-sample checks of that choice, and how well do they match? (The document answers this, Tables 2 and 5 and the ADF and Anderson-Darling columns; it should give the match for each in one line.)

---

## Data quality notes (third-party minute bars)

* Validation against NSE's daily highs, lows and opens is strong (within 5 bp on 99.3% to 100% of days; `results/data_validation/summary.csv`). The worst close difference is 51 bp on one day, consistent with the official close being a 30-minute average before August 2026.
* 38 one-minute absolute returns above 1% exist in the raw bars; all are the first bar of a day (overnight gap). The forecasting grid excludes the gap (`to_session_grid` uses the bar's own open), the replication includes it by construction, and both documents say which. Good.
* Five weekend special sessions are dropped; one official trading day (1 November 2024) has no bars; one 60-bar Muhurat session is dropped by the coverage rule. All stated.
* The closing-auction regime (3 August 2026 onward) is now documented and every affected result has a without-auction row. This was the one data feature that changed a headline, and it was caught.

---

## Numbers checked

"Found" is the value in the named result file, read with a script; "Match" is whether the document's rounding is a faithful rendering of it.

| # | Claim (document, location) | Claimed | Found | File | Match |
| --- | --- | --- | --- | --- | --- |
| 1 | REAL R0: Nifty days compared | 943 | 943 | results/data_validation/summary.csv | yes |
| 2 | REAL R0: Bank Nifty high within 5 bp | 99.4% | 99.4% | results/data_validation/summary.csv | yes |
| 3 | REAL R0: Nifty median close difference | 3.8 bp | 3.8 bp | results/data_validation/summary.csv | yes |
| 4 | REAL R0: largest Bank Nifty high or low difference | 23.3 bp | 23.3 bp | results/data_validation/summary.csv | yes |
| 5 | REAL R0: OI sum equals participant TOTAL within one contract on 9 dates | 9 dates | 9 dates, max gap 1 | results/data_validation/oi_units_check.csv | yes |
| 6 | REAL R0: index frozen 15:16 to 15:28 after 3 Aug 2026 | 96% of days | 96% | results/closing_auction/auction_evidence.csv | yes |
| 7 | REAL R1: hybrid (no GEX), next 15, all days | -6.7% | -6.71% | results/pipeline_nse/scoreboard_next15.csv | yes |
| 8 | REAL R1: hybrid (no GEX), next 30, all days | -9.0% | -9.0% | results/pipeline_nse/scoreboard_next30.csv | yes |
| 9 | REAL R1: hybrid (no GEX), next 30, expiry days | -6.6% | -6.6% | results/pipeline_nse/scoreboard_next30.csv | yes |
| 10 | REAL R1: LightGBM with GEX, next 15, expiry days | -7.2% | -7.23% | results/pipeline_nse/scoreboard_next15.csv | yes |
| 11 | REAL R1: HAR, last hour, all days | -10.8% | -10.8% | results/pipeline_nse/scoreboard_last_hour.csv | yes |
| 12 | REAL R1: VIX only, next 15, all days | +52.5% | +52.50% | results/pipeline_nse/scoreboard_next15.csv | yes |
| 13 | REAL R1: test anchors | 19,800 / 19,008 / 792 | 19,800 / 19,008 / 792 | results/pipeline_nse/scoreboard_*.csv (n) | yes |
| 14 | REAL R1: expiry days among test days | 169 of 792 | 169 of 792 | results/pipeline_nse/forecasts_last_hour.csv | yes |
| 15 | REAL R1: DM hybrid vs mcsGARCH, next 15 and 30, all days | p < 0.0001 | 1.5e-12, 0.0 | results/pipeline_nse/tests_next15.csv, tests_next30.csv | yes |
| 16 | REAL R1: DM hybrid vs mcsGARCH, expiry days | 0.0007, 0.0002 | 0.00067, 0.00024 | results/pipeline_nse/tests_next15.csv, tests_next30.csv | yes |
| 17 | REAL R1: DM hybrid vs mcsGARCH, last hour | 0.26 | 0.264 | results/pipeline_nse/tests_last_hour.csv | yes |
| 18 | REAL R1: DM LightGBM vs mcsGARCH, expiry days | 0.001, 0.0001 | 0.00116, 0.00012 | results/pipeline_nse/tests_next15.csv, tests_next30.csv | yes |
| 19 | REAL R1: DM HAR vs mcsGARCH, last hour | 0.015, 0.07 | 0.0148, 0.0697 | results/pipeline_nse/tests_last_hour.csv | yes |
| 20 | REAL R1: HAR without auction days | 4.2% (p 0.11), 7.1% (p 0.17) | 4.2%, 0.108; 7.1%, 0.167 | results/closing_auction/r1_dm_tests.csv | yes |
| 21 | REAL R1: hybrid without auction days | 6.0, 7.8, 5.5, 5.7%; p <= 0.0013 | 6.0, 7.8, 5.5, 5.7; max p 0.00128 | results/closing_auction/r1_dm_tests.csv | yes |
| 22 | REAL R1: GEX ablation for hybrid, DM p range | 0.11 to 0.87 | 0.109 to 0.874 | results/pipeline_nse/tests_*.csv | yes |
| 23 | REAL R1: LightGBM GEX ablation p | 0.13, 0.09 | 0.130, 0.092 | results/pipeline_nse/tests_next15.csv, tests_next30.csv | yes |
| 24 | REAL R1: mcsGARCH alpha, beta, t degrees of freedom | 0.049, 0.945, 5.9 | 0.0494, 0.9447, 5.95 | results/pipeline_nse/mcsgarch_full_sample.csv | yes |
| 25 | REAL R1: half-life about 2 hours | about 2 h | 118 min from alpha+beta | results/pipeline_nse/mcsgarch_full_sample.csv | yes |
| 26 | REAL R1: 13 refits, alpha range | 0.025 to 0.051 | 13 rows, 0.0254 to 0.0512 | results/pipeline_nse/mcsgarch_params.csv | yes |
| 27 | REAL R2: multi-weekly regime, whole day | 98 weeks, +11%, +1 to +22%, p 0.045 | 98, 1.1146, interval and p as stated | results/real_day_level/r2_expiry_effect_by_regime.csv | yes |
| 28 | REAL R2: Thursday single-weekly, whole day | 41 weeks, -11%, -26 to +8%, p 0.26 | 41, 0.888, as stated | results/real_day_level/r2_expiry_effect_by_regime.csv | yes |
| 29 | REAL R2: Tuesday regime, whole day and last hour | 61 weeks, +21%, p 0.005; +43%, p 0.00005 | 61, 1.211, 0.005; 1.428, 5e-5 | results/real_day_level/r2_expiry_effect_by_regime.csv | yes |
| 30 | REAL R2: Mann-Whitney Tuesday vs Thursday single | 0.012 | 0.0116 | results/real_day_level/r2_expiry_effect_by_regime.csv | yes |
| 31 | REAL R2: Tuesday weeks before auction | 48 weeks, +24% (0.005), +42% (0.00004) | 48, 23.5%, 0.0047; 42.2%, 3.8e-5 | results/closing_auction/r2_tuesday_split.csv | yes |
| 32 | REAL R3: US rule GEX and interaction | -0.38 (0.030), +0.39 (0.11) | -0.3824, 0.0297; 0.3935, 0.113 | results/real_day_level/r3_gex_regressions.csv | yes |
| 33 | REAL R3: Pro rule | -2.07 (0.033), +4.93 (0.002) | -2.068, 0.033; 4.929, 0.00165 | results/real_day_level/r3_gex_regressions.csv | yes |
| 34 | REAL R3: Client mirror | -2.35 (0.013), +3.65 (0.002) | -2.353, 0.0133; 3.652, 0.0023 | results/real_day_level/r3_gex_regressions.csv | yes |
| 35 | REAL R3: R squared and days | 0.467 to 0.469..0.474; 937 days | 0.4672; 0.4689 to 0.4735; 937 | results/real_day_level/r3_gex_regressions.csv | yes |
| 36 | REAL R4: Mann-Whitney p, max-OI and max-pain | 0.67, 0.12 | 0.673, 0.116 | results/real_day_level/r4_pinning.csv | yes |
| 37 | REAL R5: slope all days; expiry days | -0.029 (0.015); -0.037 (0.019) | -0.0294, 0.0153; -0.0374, 0.0193 | results/momentum_gex/regressions.csv | yes |
| 38 | REAL R5: days with GEX < 0 by rule | 107 / 463 / 38 | 107 / 463 / 38 | results/momentum_gex/regressions.csv | yes |
| 39 | REAL R5: Pro interaction, three specifications | -0.068 (0.0002), -0.069 (0.0006), -0.050 (0.003) | as stated | results/momentum_gex/regressions.csv | yes |
| 40 | REAL R5: Client mirror, three specifications | -0.072 (0.00003), -0.069 (0.001), +0.003 (0.95) | as stated | results/momentum_gex/regressions.csv | yes |
| 41 | REAL R5: US rule, three specifications | +0.013 (0.59), +0.002 (0.95), -0.001 (0.97) | as stated | results/momentum_gex/regressions.csv | yes |
| 42 | REAL R5: slope after positive GEX | +0.011 (0.29) | +0.0106, 0.291 | results/momentum_gex/regressions.csv | yes |
| 43 | REAL R5: stability p by regime and half | 0.0005 to 0.17; 0.03, 0.01 | 0.0005 to 0.17; 0.033, 0.0105 | results/momentum_gex/stability.csv | yes |
| 44 | REAL R5: follow-the-day payoff | -2.1 bp, t -2.3 | -2.10, -2.35 | results/momentum_gex/gamma_states.csv | yes |
| 45 | REAL R5: without auction days | -0.030 (0.017); -0.072 (0.0001); -0.053 (0.001) | -0.0303, 0.0165; -0.0719, 7.8e-5; -0.0526, 0.00127 | results/closing_auction/r5_without_auction.csv | yes |
| 46 | PINNING 4.1: pooled distance, expiry vs other | 11.52 vs 12.62, -1.10, -2.25 to 0.03, p 0.054 | as stated (p 0.0538) | results/pinning/t1_clustering.csv | yes |
| 47 | PINNING finding 2: Thursday single-weekly regime | 8.30 vs 12.82, -4.52, p 0.0006 | as stated | results/pinning/t1_clustering.csv | yes |
| 48 | PINNING: Tuesday regime | 14.55 vs 13.37, p 0.27 | 14.55, 13.37, 0.2676 | results/pinning/t1_clustering.csv | yes |
| 49 | PINNING 4.2: Thursday vs Tuesday regime gaps differ | p 0.0008 | 0.0008 | results/pinning/t1_regime_heterogeneity.csv | yes |
| 50 | PINNING: Bank Nifty while weekly | 20.26 vs 24.52, -4.26, p 0.0076 | as stated | results/pinning/t1_clustering.csv | yes |
| 51 | PINNING: Holm-adjusted gamma conditioning | 0.896, 0.523, 0.914 | as stated | results/pinning/t5_gamma_conditioning.csv | yes |
| 52 | PINNING: minimum detectable effect | 2.60 to 3.09 points | 2.60 to 3.09 | results/pinning/t6_power_mde.csv | yes |
| 53 | PINNING: same weekday, different status | -3.40 (p 0.0035); 2.26 (p 0.039) | as stated | results/pinning/t7_robustness.csv | yes |
| 54 | PINNING: close proxy error | 1.03 points median, 96% within | 1.03, 96% | results/pinning/t4_close_proxy_check.csv | yes |
| 55 | REPL finding 1: after-sample Nifty N | 21,239 | 21,239 | results/replication_nse/table1_descriptive.csv | yes |
| 56 | REPL Table 1: inferred-window Nifty maximum | 0.006221 (printed 0.006221) | 0.006221108412 | results/replication_nse/table1_descriptive.csv | yes |
| 57 | REPL Table 1: inferred-window Nifty minimum | -0.008534 (printed -0.008585) | -0.008534 | results/replication_nse/table1_descriptive.csv | yes (file); differs from printed, see M7 |
| 58 | REPL Table 2: half-lives before and after, Nifty and Bank Nifty | 358.6 / 72.4 / 162.0 / 38.0 min | as stated | results/replication_nse/table2_persistence.csv | yes |
| 59 | REPL: likelihood-ratio p for the change | 5.6e-5, 1.9e-5 | 5.6e-5, 1.9e-5 | results/replication_nse/table2_persistence_change.csv | yes |
| 60 | REPL: window sensitivity of the half-life | 318 to 370; 138 to 166 | as stated | results/replication_nse/table2_window_sensitivity.csv | yes |
| 61 | REPL Tables 3 and 4: cross-index shares and net | 49.0 / 38.9; -11.2 / -1.2 | as stated | results/replication_nse/tables34_connectedness.csv | yes |
| 62 | REPL Table 6: Nifty Thursday expiry | +49.3%, p 0.0102 / 0.000048 / 0.002 | as stated | results/replication_nse/table6_volatility.csv | yes |
| 63 | REPL: best of 348 windows and score | 14 Nov 2022 to 10 Feb 2023, 0.045 | rank 1, 0.0447; 348 rows | results/replication_nse/fingerprint_search.csv | yes |
| 64 | REPL: best window starting on or after 1 Nov 2023 | rank 217, score 6.21 | rank 217, 6.207 (29 Nov 2023 to 29 Feb 2024) | results/replication_nse/fingerprint_search.csv | yes |

Beyond these 64 rows I also checked every remaining cell of the R1 scoreboard (42 cells), the full R2 table, the Table 1, 2, 5 and 6 cells of the replication, and the printed-number checks in `paper_printed_checks.csv` (Bank Nifty before-sample N implied 22,680; Table 4 first-hour rows summing to 101; 71.97 against 70.97; +65% against 61%). All agree with the documents. I found no number in any of the three documents that is mis-transcribed from its result file.
