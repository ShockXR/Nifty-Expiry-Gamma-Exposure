# Methodology validation: Nifty expiry gamma exposure (prototype A)

Written 3 November 2026 by validation agent 2 (an AI agent), at commit adbc553 plus the working-tree edits listed in `reports/findings/A2_A.csv`. Labels: REAL (NSE market data), SYNTHETIC (simulated), PAPER (printed in Chhimwal, Pandey and Pandey 2025), OUR CHECK (recomputed by this validation).

## Summary for a beginner

1. The project asks whether data on option dealers' positions (gamma exposure, GEX) help forecast how wildly Nifty will move in the next 15 to 60 minutes, especially on option-expiry days.
2. The honest answer so far is no for GEX itself. A machine-learning correction to the paper's volatility model (mcsGARCH) does forecast better, but the gain comes from recent volatility and time of day, not from option positions.
3. I re-derived the 20 load-bearing formulas from the code (Black-76 gamma, GEX, QLIKE, Diebold-Mariano, mcsGARCH and so on) and checked them with independent numeric tests: 32 of 32 checks pass (OUR CHECK, `reports/checks/equation_vs_code_check.py`).
4. I also recomputed the headline gains straight from the committed forecast files and they match the README (5.80%, 7.65%, 6.36% against our mcsGARCH; 4.43%, 6.64%, 6.16% against the paper's specification).
5. What I fixed: a stale test count, an over-broad "five sign rules" line, "same information" wording that the evidence does not support, and an uncaveated event-study headline. What stays open is listed in section 7.

## 1. Problem

Does public NSE option data (open interest by strike and by participant group, prices) carry information about the next 15, 30 and 60 (last hour) minutes of Nifty variance beyond what a good time-series model already knows? Secondary questions: does dealer positioning explain the end-of-day reversal, expiry-day volatility and pinning; does the paper's before/after result replicate. Neither prototype is a trading system. The metrics are forecast loss (QLIKE) and tests, not Sharpe ratios.

## 2. Assumptions (those that matter most)

| Assumption | Where | Why it matters |
| --- | --- | --- |
| Dealers' net position in each option is a fixed share of open interest (sign rule). Public data never show it. | `gex.py` module docstring | GEX is only as good as the sign rule; three rules are reported side by side and none helps forecasts. |
| Black-76 with a flat risk-free rate (6.5%) is adequate for weekly index options. | `greeks.py`, `docs/math/01` | Expiry-day options have very small T, so gamma is huge and sensitive to implied volatility. |
| Realized variance of 1-minute log returns is a good enough target. | `features.py` (y = sum of squared returns over the window) | QLIKE ranks forecasts correctly under a noisy target (Patton 2011); bid-ask noise at the close is a known issue (`docs/EVENT_STUDY.md`). |
| Daily average losses are the unit for tests (n = number of test days), not the anchors within a day. | `evaluation.py` | Anchors within a day share the same shock; treating them as independent would overstate significance. |
| Walk-forward: forecasts for day t use only information published by the close of day t-1. | `features.py` (panel row prev = t-1), `docs/math/03` section 4 | Look-ahead would invalidate every result. |

## 3. Methodology (pipeline)

Inputs: NSE F&O bhavcopies, participant-wise open interest, India VIX, and free 1-minute index bars checked day by day against NSE official values (REAL, 942 trading days, 2 January 2023 to 2 November 2026). Raw files are not committed; derived results are (`results/`).

1. Implied vol from settlement prices, Black-76 greeks, GEX per strike under each sign rule (`greeks.py`, `gex.py`).
2. Forecast ladder on 3 horizons (next 15 min, next 30 min, last hour): naive, HAR, mcsGARCH, linear, LightGBM, and the hybrid (mcsGARCH forecast multiplied by a LightGBM correction learnt from 41 inputs) (`forecast.py`, `volatility.py`).
3. Walk-forward evaluation: first 150 days training only, refit every 21 days, hyperparameters chosen on the last 20% of each training window by QLIKE; 792 test days (`docs/math/03` section 4).
4. Scoring: mean QLIKE and MSE, percent gain against mcsGARCH, Diebold-Mariano on daily losses, Model Confidence Set, Hansen SPA over 31 challengers (post hoc).
5. Every hypothesis test is listed in `results/test_ledger.csv` (28,515 tests, 37 studies) with Holm and Benjamini-Yekutieli corrections (OUR CHECK: 3,544 survive the project-wide BY correction, matching `docs/TEST_LEDGER.md`).

Hyperparameters (stated in code, not tuned for the headline): training minimum 150 days, refit block 21 days, tuning on last 20% of training window, trimmed-QLIKE trim 1%, DM lag floor(4 (n/100)^(2/9)).

## 4. Implementation: equation versus code

Line numbers refer to the working tree at the time of checking.

| # | Claim (document) | Code | Verdict | Evidence |
| --- | --- | --- | --- | --- |
| 1 | Black-76 gamma = e^(-rT) phi(d1) / (F sigma sqrt T) (`docs/math/01` s.2) | `greeks.py:56-60` | MATCH | Second finite difference of price: 1.243122e-03 vs code 1.243123e-03, call and put (OUR CHECK) |
| 2 | Black-76 price, parity, delta | `greeks.py:41-54` | MATCH | C - P = e^(-rT)(F - K) to 1e-8; delta equals central difference |
| 3 | Spot gamma = gamma_F (dF/dS)^2 with F = S e^(carry T) | `greeks.py:93-99` | MATCH | Finite difference in S: 1.258407e-03 vs 1.258408e-03 |
| 4 | GEX_k = Gamma OI L S^2 0.01 (`docs/math/01` s.4; `gex.py` header) | `gex.py:93-116` | MATCH | Hand formula equal to 1e-10. First-order definition (delta change per 1% move times units times S) agrees to 2.6e-4; a full +/-0.5% bump differs by -2.5% for a 3-day at-the-money option because gamma changes over 1%. The docs say "first order", which is correct |
| 5 | Sign rules: US convention (+1 calls, -1 puts), NSE Pro participants, mirror of NSE Clients, long-all, short-all (`gex.py` docstring) | `gex.py:48, 52-90`, pipeline use `gex.py:218-224`, `features.py:117` | MATCH | All five coded and tested. The pipeline and every result table use only the first three. README said "five sign rules" without saying so: edited |
| 6 | Participant weight = (long - short) / total contracts, clipped to [-1, 1]; mirror flips sign | `gex.py:57-75` | MATCH | Hand example reproduced |
| 7 | Gamma flip = root of net GEX(s) nearest spot, grid then Brent | `gex.py:141-175` | MATCH (by reading only) | Not numerically re-derived by me; covered by `tests/test_gex.py` |
| 8 | QLIKE(y, f) = y/f - ln(y/f) - 1 (`docs/math/03` s.3) | `evaluation.py:19-23` | MATCH | Zero and minimised at f = y, scale-free, penalises under-prediction more: 3 checks. Both arguments floored at 1e-16 (flat-bar days, see `docs/DATA_QUALITY_FLAT_BARS.md`) |
| 9 | DM on daily mean losses, Bartlett long-run variance, Harvey-Leybourne-Newbold factor, Student t with n - 1 df, lag floor(4 (n/100)^(2/9)) | `evaluation.py:39-70` | MATCH | Independent reimplementation gives identical statistic and p. Empirical size at nominal 5%, n = 792, 3,000 simulations: 5.2% (iid differentials), 5.6% (AR(1) = 0.3) (OUR CHECK, SYNTHETIC) |
| 10 | Losses averaged within day before testing | `evaluation.py:32-36` | MATCH (by reading) | n in all tests is days (792), confirmed in recomputation below |
| 11 | mcsGARCH: r = mu + sqrt(h s q) z; q follows GARCH(1,1) across day boundaries; standardised-t likelihood; omega = e^a, persistence = 0.99999 logit^-1(b), nu = 2.05 + e^d (`docs/math/02`) | `volatility.py:111-146, 257-376` | MATCH | q recursion equals filter form; t-density integrates to 1 with unit variance; parametrisation line by line |
| 12 | Multi-step forecast V(H) = h sum_k s_(i+k) E[q_(i+k)], E[q] = qbar + (a+b)^k (q - qbar) | `volatility.py:379-424` | MATCH | Monte Carlo of the recursion (200,000 paths): 43.7813 vs analytic 43.7868 (OUR CHECK, SYNTHETIC) |
| 13 | Daily variance h_t from a log-HAR with weekly and monthly means over days t-5..t-1 and t-22..t-1 | `volatility.py:427-456`, `features.py:211-212` | MATCH | Regressor construction reproduced |
| 14 | Realized variance, bipower variation (pi/2 scaling), Rogers-Satchell | `realized.py:23-60` | MATCH, with a caveat | RV and BV unbiased on simulated Brownian paths. Rogers-Satchell is unbiased only under continuous monitoring: mean over true variance is 0.788, 0.929, 0.991 with 60, 600, 6,000 sub-steps per bar. Minute bars from tick data are close to the fine case but the doc's "unbiased" needs that qualifier |
| 15 | Diebold-Yilmaz total connectedness = (100/N) sum of off-diagonal row-normalised GFEVD; paper totals are 2x for N = 2 | `connectedness.py:69-80, 143-157` | MATCH | Independent VAR(1) GFEVD gives 25.8909 = code; no-dynamics cross share rho^2/(1+rho^2) = 41.94% at rho = 0.85 matches the doc |
| 16 | Information timing: day t forecasts use day t-1 close data | `features.py:209-214` | MATCH | `gp = panel.iloc[prev]`, `ipv(..., prev)` |
| 17 | Log-scale models are multiplied by Duan's smearing factor | `forecast.py:148-149` | MATCH (by reading) | mean(exp(residual)) |
| 18 | Expiry calendar: Thursday, then Tuesday from 1 September 2025, roll back on holidays | `expiry_calendar.py:30-75` | MATCH | Spot checks on 28 August and 2 September 2025 and a holiday roll-back |
| 19 | Target y = sum of squared 1-minute returns over the horizon | `features.py:243` | MATCH | By reading |
| 20 | README/pitch headline gains | `results/pipeline_nse_v2/forecasts_*.csv`, `results/p116_benchmark_as_printed/daily_losses_nifty_*.csv` | MATCH | Recomputed: hybrid vs our mcsGARCH 5.7988%, 7.6493%, 6.3626%; vs paper specification 4.43%, 6.64%, 6.16%; 792 test days; DM one-sided p 1.8e-09, 3.6e-15, 1.0e-04 vs our version (OUR CHECK, REAL) |

## 5. Experimental evidence (where the numbers live)

All numbers are REAL unless marked. Pointers are to committed files.

* Hybrid vs paper-specification mcsGARCH: 4.4% (15 min), 6.6% (30 min), 6.2% (last hour), `results/p116_benchmark_as_printed/`. Like-for-like (hybrid rebuilt on that specification): 5.1% and 7.1%, `results/p118_hybrid_on_paper_mcsgarch/`.
* Where the gain comes from: recent realized volatility 34% and 38%, time of day 23% and 25%, option-implied, VIX and calendar inputs add nothing; a hybrid that sees only mcsGARCH inputs gets 6.1% and 8.0%, `results/p106_gain_decomposition/`. Both option groups removed together: 0.6 points, not significant.
* Five plain fixes to mcsGARCH recover 118% and 108% of the hybrid's gain and the hybrid is not better than the fixed model, `results/p109_mcsgarch_extensions/`. The fixes were chosen after seeing the days, so a forward test is registered from 4 November 2026 (`docs/prereg/P112_FORWARD_MCSGARCH_FIXES.md`).
* Replication on other indices: Bank Nifty 3.9% and 5.0%, Fin Nifty 4.9% and 6.1%, pooled 4.7% and 6.0%, counted as about two independent replications (`docs/P107_POOLED_THREE_INDEX.md`).
* GEX features add nothing out of sample on Nifty under any of three sign rules (`docs/MODEL_ITERATIONS_REAL.md`).
* Paper replication: 21,239 Nifty returns reproduced; mcsGARCH alpha 0.0498 and beta 0.9407 against printed 0.0494 and 0.9410 (PAPER); the stated "before" window does not reproduce, a 64-day window starting 14 to 23 November 2022 does (`docs/REPLICATION_NSE.md`, `docs/P108_BLIND_WINDOW.md`).
* Number tracing (OUR CHECK, rerun after my edits): 678 registry entries, 664 MATCH, 0 MISMATCH, 0 TEXT_MISSING, 0 SOURCE_MISSING, 14 UNTRACED (design constants and derived quantities, each with a note). The pitch has 0 numbers outside the registry.
* Tests: 715 collected (`pytest --collect-only`), 708 passed and 7 skipped without raw data (coordinator's run at adbc553).

## 6. Results in one table

| Question | Answer | Strength |
| --- | --- | --- |
| Can the paper's mcsGARCH be beaten at 15 and 30 minutes? | Yes, by about 4% to 7% | Strong (survives project-wide BY, replicates on 2 more indices) |
| Does GEX help forecasts? | No | Strong negative result (three sign rules, registered ablation) |
| Is the gain due to option information? | No, mainly learnt correction from recent volatility and time of day; option inputs under 1% of loss | Registered, but one data set |
| Is the gain more than "fix mcsGARCH properly"? | Not shown: five pre-written fixes recover 108% to 118% | Forward test pending |
| Does the paper's persistence fall come from the expiry change? | NOT SETTLED | Placebo cannot separate it for Nifty |
| End-of-day reversal after negative Pro GEX | In-sample only: Holm over 78 headline tests p = 0.016, not over all 742 | Weak |
| Does the paper's "before" window replicate? | Only on a window one year earlier, start 14 to 23 November 2022 | Two AI searches differ by about a week |

## 7. Limitations and failure scenarios

* Single market, single sample (2023 to 2026), one data vendor for minute bars; raw data not in the repo.
* The hybrid corrects our own mcsGARCH implementation, which adds an expiry-day curve and smoothing and is slightly worse than the paper's specification on Nifty. Headlines therefore lead with the paper-specification comparison.
* Post hoc analyses (information-matched benchmark 4.8% and 5.9%, economic value about 0.04% of wealth a year) follow AI referee reports and fail the project-wide correction.
* All reviews and cross-checks (INDEPENDENT_REVIEW, REKEY_CHECK, referee responses) were done by AI agents, not people. The README says so.
* Failure scenarios: dealer sign wrong for India (the main open problem); a change in NSE rules (the closing auction from 3 August 2026 already changed one result); flat-bar days from data outages (handled, see `docs/DATA_QUALITY_FLAT_BARS.md`); connectedness intervals that rest on few independent periods; heavy multiple testing (28,515 tests, only 3,544 survive the project-wide correction).
* Open items for a human or another agent: A2-A-004 (generator script text for the ledger), A2-A-006 (dated test counts in `docs/REPRODUCIBILITY.md`), A2-A-007 (Rogers-Satchell docstring qualifier).
