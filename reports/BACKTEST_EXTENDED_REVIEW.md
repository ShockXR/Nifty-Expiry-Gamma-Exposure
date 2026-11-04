# Review of experiments/backtest_extended.py (independent check)

Done by an AI agent (Claude, Agent F) on 4 November 2026, about 02:20 to 02:30 UTC. I reviewed and tested the statistics; I did not rewrite the script or change any committed number. Inputs to the spot checks are the committed forecast and loss files (**REAL DATA**). The size checks use simulated data (**SYNTHETIC**). Scratch code: `/tmp/claude-0/-home-user/e79a6425-7016-5765-9dd9-652f4c629e5e/scratchpad/agentF/review/` (`size.py`, `spot.py`).

## Verdict: VERIFIED (no defect that changes a conclusion; no fix made)

Every method matches its textbook definition. The tests had about the right false-alarm rate (size) on simulated data. All three spot numbers reproduced with independent code. The deviations below are small or already disclosed in `docs/BACKTEST_EXTENDED.md`; none can change a conclusion of the document. There was no one-line bug, so nothing was edited.

## 1. Method against textbook

| Item | Textbook | What the code does | Deviation? | Can it change a conclusion? |
| --- | --- | --- | --- | --- |
| Stationary bootstrap (Politis and Romano 1994) | Blocks start at random points, geometric lengths with mean 1/q, wrap around | Same (`stationary_bootstrap_indices`); measured mean block 9.98 for mean 10 (SYNTHETIC) | None | No |
| SPA statistic (Hansen 2005) | T = max(0, max_k sqrt(n) dbar_k / omega_k), d = benchmark loss minus model loss | Same | None | No |
| SPA studentisation | omega from the stationary-bootstrap kernel, weights (n-i)/n (1-q)^i + i/n (1-q)^(n-i) | Same (`sb_variance`). The sum stops once a weight is below 1e-12 | Negligible truncation; equals 1.016 on iid unit-variance data (SYNTHETIC) | No |
| SPA recentring, lower / consistent / upper | g_l(x) = max(x, 0); g_c keeps x only if x >= -sqrt(omega^2 2 log log n / n), else 0; g_u(x) = x | Same, written in t units (threshold sqrt(2 log log n)); p_lower <= p_consistent <= p_upper as required | None | No |
| SPA block length and seed | Free choice | Mean block 10 days, 5,000 draws, seed 9700001, recorded in `spa.csv` | Not tuned; a p of 0.000 means below 1/5000 (disclosed) | No |
| White's reality check | Not studentised, recentred | Same | None | No |
| Giacomini and White (2006) | n Zbar' Omega^-1 Zbar, chi-square(q), Z_t = h_{t-1} d_t; for one-step forecasts Omega = mean of Z Z' | Reuses `referee_forecast_gw.gw_test`, which does exactly this; the slope Wald test uses Newey-West | None in the formula. **But** the tercile cut points (VIX, RV, distance to flip) come from all 792 test days, so those instruments are not strictly known at t-1 | No. These are fixed thresholds, not fitted values; every GW p in the doc is large |
| Regime cells | Description | Gains per cell with the repo's DM test; terciles over the full test sample | Same look-ahead in the cell edges, harmless for a description | No |
| Mincer-Zarnowitz with HAC, joint (0, 1) Wald | y = a + b f + e, H0: a = 0, b = 1 | OLS; scores summed within each day, then Newey-West across days; chi-square(2) Wald | None. The doc correctly says the log-scale joint test is not a calibration test (Jensen's inequality) | No |
| Encompassing | y - f_m = c + w (f_h - f_m) + e | Same, HAC as above | None | No |
| QLIKE-optimal combination weight | Weight chosen on a sample, then evaluated out of sample | Grid weight chosen **on the same 792 days** it is then tested on (DM against the hybrid) | In-sample optimum, so its gain is biased upward. The doc's figure, 0.08% (p 0.22), is already insignificant | No; label it "in-sample weight" |
| Patton (2011) family | b = -2 QLIKE, b = -1 f - y + y log(y/f), b = 0 (y - f)^2 / 2, general formula otherwise | Same. b = -2 equals QLIKE exactly (max difference 0); b = 0 equals (y - f)^2 / 2 (relative difference 6e-10); the general formula is continuous into both special cases (SYNTHETIC check) | None | No |
| Rolling 63-day gain, losing runs, drawdown | Moving means, runs of days where the hybrid loses | Same; a tie counts as a win | Negligible | No |

## 2. Size checks (SYNTHETIC)

All use the script's own functions: `spa_test` and `hac_ols` + `wald`.

| Check | Setup | Seed | Rejection rate at 5% |
| --- | --- | --- | --- |
| SPA, independent models | 200 worlds, n = 792, benchmark + 32 models, iid lognormal losses exp(0.5 z) with equal means, 1,000 draws, mean block 10 | 20261104 | consistent 0.06, lower 0.07, upper 0.06 (reality check 0.03) |
| SPA, correlated models | Same, 80% common factor in log losses (closer to real ladders) | 20261105 | consistent 0.06, lower 0.06, upper 0.06 (reality check 0.05) |
| MZ joint (0, 1), variance scale | 200 worlds, 792 days x 8 anchors, f = true conditional mean (day level x anchor noise), y = f x chi-square(4)/4 | 20261106 | 0.055 |
| MZ joint (0, 1), log scale, same worlds | Calibrated in variance, so a log intercept below 0 is expected | 20261106 | 1.00 (as the doc says: not a calibration test on the log scale) |

With 200 worlds the Monte Carlo standard error is about 1.6 percentage points, so 0.055 to 0.07 is consistent with 5%. Each check took under 4 minutes.

## 3. Spot checks (REAL DATA, own code)

| Number in `docs/BACKTEST_EXTENDED.md` | Own recomputation | Status |
| --- | --- | --- |
| 15-minute hybrid gain over mcsGARCH, 5.80% (792 days) | 5.799% from `results/pipeline_nse_v2/forecasts_next15.csv` (daily mean QLIKE, then mean over days) | VERIFIED |
| Previous-day VIX low tercile, 15 minutes, 6.10% on 264 days | 6.096% on 264 days (VIX of `daily_market.csv` shifted one day, lowest third) | VERIFIED |
| SPA, hybrid as benchmark, 15 minutes, p = 0.923 (32 alternatives, statistic 0.63) | p = 0.929, statistic 0.61, 32 alternatives. Own bootstrap (2,000 draws, seed 12345), own Bartlett long-run variance with lag 20 instead of Hansen's kernel, so a small difference is expected | VERIFIED |

## 4. Suggested fixes for the author (none required for the conclusions)

1. Call the QLIKE-optimal combination weight "chosen on the test days (in-sample)" in `docs/BACKTEST_EXTENDED.md`, or choose it on an earlier window.
2. Note that the tercile cut points in the regime table and the GW instruments use the full test sample. If strictness matters, use expanding-window terciles.
3. Optional: record the Monte Carlo floor of the SPA p-values in `spa.csv` (1/draws), not only in the text.
