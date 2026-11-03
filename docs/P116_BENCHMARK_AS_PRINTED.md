# P116: how large is the hybrid's gain over mcsGARCH as the paper specifies it?

Study P116, run by an AI agent (Claude) working for Shreyan Thanki on 3 November 2026.
**All numbers below are REAL DATA** (Nifty, Bank Nifty and Fin Nifty, NSE one-minute index data as in the
headline runs) unless marked SYNTHETIC. Every number comes from a file in `results/p116_benchmark_as_printed/`
(file named next to each table) or, where marked, from an earlier committed results file.

* Registered plan: `docs/prereg/P116_BENCHMARK_AS_PRINTED.md`, commit `19c5486cd39e9944764ac9a1ea7c3ab3bbb7b63b`
  (committed 10:57:48 UTC and pushed at about 10:58 UTC, before any forecast of a new benchmark variant was made on
  real data and before any loss of this study was computed).
* Code (all in the plan commit, unchanged since): `experiments/p116_benchmark_as_printed.py`; one new code path in
  `experiments/p109_mcsgarch_extensions.py` (optional switches in a variant's spec, defaults as committed); new
  functions at the end of `src/niftygex/volatility.py` (`arma11_conditional_mean`, `arma11_mean_square_forecast`).
  Tests (SYNTHETIC only): `tests/test_p116_benchmark_as_printed.py`. One post hoc descriptive script was added after
  the run (`experiments/p116_posthoc_diurnal_curves.py`, Section 13).
* Every p-value of the study: `results/p116_benchmark_as_printed/tests.csv` (77 rows: 4 primary, 16 secondary, 57
  exploratory).

Units. A **gain** of model m over a benchmark b is 100 x (L_b - L_m) / L_b, in % of the benchmark's mean daily
QLIKE (the same unit as the "points of mcsGARCH's QLIKE" of P106 and P109); positive means m is better. **Delta** is
the hybrid's gain over `mcs_cpp` minus its gain over `mcs_ours`, in percentage points: negative means the headline
(measured against `mcs_ours`) is larger than the gain over the paper's specification.

## 1. The answer in plain words (REAL DATA)

Our hybrid forecasts how much the Nifty will move in the next 15 or 30 minutes. Its headline gain, 5.80% and 7.65%
lower QLIKE than mcsGARCH, is measured against **our version** of mcsGARCH (`mcs_ours`), which makes three choices
the anchor paper does not make: a separate time-of-day curve for expiry days, a 5-minute smoothing of that curve,
and a zero expected return. This study refitted mcsGARCH **as the paper specifies it** (`mcs_cpp`: one time-of-day
curve for all days, no smoothing, an ARMA(1,1) expected return, Student t errors), with exactly the same
walk-forward, days, anchors and targets, and every mix of the three choices.

**Does the hybrid still beat mcsGARCH as the paper specifies it? Yes, clearly.**

* Against `mcs_cpp` the hybrid is **4.43% better at 15 minutes and 6.64% better at 30 minutes** (one-sided
  Diebold-Mariano p below 0.001 at both, Holm-adjusted below 0.001), and 6.16% better in the last hour (Holm p below
  0.001).

**Does our extension make the benchmark better or worse? On Nifty, worse.**

* `mcs_ours` has **1.45% higher QLIKE than `mcs_cpp` at 15 minutes and 1.09% higher at 30 minutes** (in % of
  `mcs_cpp`'s QLIKE; two-sided DM, Holm-adjusted p 0.005 and 0.014). Our three choices made the benchmark weaker,
  not stronger.
* So the headline is larger than the gain over the paper's own specification: Delta is **-1.37 points at 15
  minutes** (95% interval -2.15 to -0.58) and **-1.00 points at 30 minutes** (-1.77 to -0.23).
* The registered verdict is **"headline overstated"** at both horizons: H2 is significant after Holm, `mcs_cpp`
  has the lower loss, and the change is larger than the registered 0.5-point margin.

**Which of our three choices is to blame?** Two of them, about equally; the third does nothing.

* The **separate expiry-day curve** costs 0.84 points at 15 minutes and 0.39 at 30 minutes (Shapley split of the
  change, points of `mcs_cpp`'s QLIKE). Its damage is on expiry days, where `mcs_ours` is 3.93% and 2.01% worse than
  `mcs_cpp` (Holm p 0.001 and 0.038): a curve estimated from about one day in five is too noisy.
* The **5-minute smoothing** costs 0.62 and 0.69 points. It spreads the sharp one-minute spike at 15:00 into 14:58
  and 14:59, so our benchmark forecasts the 14:45 window too high and the 15:00 window too low (post hoc,
  descriptive, Section 4). At the 09:15 forecast, by contrast, the smoothing happens to help.
* The **zero mean instead of ARMA(1,1)** changes the loss by 0.01 points or less: the fitted AR and MA terms nearly
  cancel.

**Other indices, and P109.** On **Bank Nifty** the result goes the other way: our version is better than the
paper's (by 0.84% and 1.23%, Holm p 0.003 and below 0.001), so there the headline (3.92% and 5.03%) understates the
gain over the paper's specification (4.73% and 6.20%). On **Fin Nifty** our version is worse by 0.91% and 0.97%, but
not significantly after Holm (0.059 and 0.074), so the outcome there is "inconclusive". The hybrid beats `mcs_cpp`
on every index at both horizons. P109's combination beats `mcs_cpp` by 5.48% and 7.25% on Nifty (Holm p below
0.001).

**What it means.** The project's benchmark should be described as weaker than the paper's own specification on
Nifty, and the Nifty headline should be quoted against the paper's specification: **4.43% and 6.64%** (with 5.80%
and 7.65% against our earlier version as a comparison). The hybrid's advantage is real and significant against
either benchmark; when the benchmark is the paper's model it is about a quarter smaller at 15 minutes (1.37 of
5.80 points, 24%) and about an eighth smaller at 30 minutes (1.00 of 7.65 points, 13%) than reported. Section 10 gives the wording.

## 2. What was registered (in short)

* **Models.** `mcs_ours` (the committed benchmark, rebuilt and checked to be identical to the committed
  forecasts); `mcs_cpp` (`McsGarch(mean="arma11", dist="t", expiry_diurnal=False, smooth=1)`, the replication's
  model of Table 2); the six other combinations of the three switches (a 2 x 2 x 2 factorial: `ours_no_expiry`,
  `ours_no_smooth`, `ours_arma` switch one choice of `mcs_ours` to the paper's; `cpp_expiry`, `cpp_smooth`,
  `cpp_zero_mean` switch one choice of `mcs_cpp` to ours); `mcs_cpp_varonly` (the `mcs_cpp` fit without the
  squared mean in its forecast, exploratory); the committed hybrid (seed set 7) and P109's committed combination,
  unchanged. The same daily AR(1)-EGARCH(1,1) level, Student t GARCH(1,1), cross-day recursion and walk-forward for
  every model.
* **The forecast with an ARMA(1,1) mean** is the model's expectation of the target, the sum of squared one-minute
  returns: the usual variance part plus the expected squared conditional mean (plan Section 1).
* **Design.** The committed walk-forward through P109's driver: the first 150 days for training only, then a refit
  on all earlier days before every block of 21 trading days. Nifty 792 test days (10 August 2023 to 2 November
  2026), Bank Nifty 660, Fin Nifty 604; anchors every 15 minutes; the last hour at 14:30.
* **Primary tests** (4 rows, Holm over all four), Nifty, 15 and 30 minutes: H1, the hybrid beats `mcs_cpp`
  (one-sided DM); H2, `mcs_ours` and `mcs_cpp` differ (two-sided DM). DM on daily mean QLIKE with Newey-West
  variance, HLN factor and Student t, as in P109.
* **Decision rule** at each horizon: "headline understated" (H2 Holm p below 0.05, `mcs_cpp` worse), "headline
  overstated" (H2 Holm p below 0.05, `mcs_cpp` better), each noted if |Delta| is at most 0.5 points; "no material
  difference" (H2 not significant and |Delta| at most 0.5 points); "inconclusive" (H2 not significant and |Delta|
  above 0.5). The 0.5-point margin is wider than the whole seed-to-seed range of the hybrid's gain in P99.
* **Secondary families** (Holm within each): S1 last hour (2), S2 expiry days (4), S3 P109's combination against
  `mcs_cpp` (2), S4 Bank Nifty (4), S5 Fin Nifty (4). **Exploratory** (no adjustment): E1 one switch from
  `mcs_ours`, E2 one switch from `mcs_cpp`, E3 the squared-mean term, E4 the 09:15 anchor and the rest, E5
  non-expiry days.

## 3. The main result (REAL DATA, Nifty, 792 test days)

From `losses.csv`, `tests.csv` and `verdict.csv`. Mean daily QLIKE (lower is better), and gains in % of each
benchmark's QLIKE.

| Model | 15 min: QLIKE | gain vs `mcs_ours` | gain vs `mcs_cpp` | 30 min: QLIKE | gain vs `mcs_ours` | gain vs `mcs_cpp` | last hour: QLIKE | gain vs `mcs_ours` | gain vs `mcs_cpp` |
|---|---|---|---|---|---|---|---|---|---|
| **`mcs_ours`** (committed benchmark) | 0.2845 | 0 | -1.45 | 0.2176 | 0 | -1.09 | 0.2395 | 0 | -0.22 |
| **`mcs_cpp`** (the paper's specification) | 0.2804 | +1.43 | 0 | 0.2152 | +1.08 | 0 | 0.2390 | +0.22 | 0 |
| `mcs_cpp_varonly` (no squared mean) | 0.2805 | +1.43 | -0.00 | 0.2152 | +1.07 | -0.00 | 0.2390 | +0.20 | -0.02 |
| `ours_no_expiry` (one curve) | 0.2826 | +0.69 | -0.75 | 0.2169 | +0.31 | -0.77 | 0.2401 | -0.24 | -0.46 |
| `ours_no_smooth` (no smoothing) | 0.2832 | +0.47 | -0.98 | 0.2162 | +0.61 | -0.47 | 0.2386 | +0.39 | +0.17 |
| `ours_arma` (ARMA mean) | 0.2845 | -0.01 | -1.46 | 0.2175 | +0.01 | -1.08 | 0.2392 | +0.13 | -0.08 |
| `cpp_expiry` (expiry curve) | 0.2832 | +0.47 | -0.98 | 0.2162 | +0.62 | -0.46 | 0.2383 | +0.52 | +0.31 |
| `cpp_smooth` (smoothing) | 0.2826 | +0.68 | -0.76 | 0.2169 | +0.32 | -0.76 | 0.2398 | -0.11 | -0.33 |
| `cpp_zero_mean` (zero mean) | 0.2804 | +1.44 | +0.01 | 0.2152 | +1.07 | -0.00 | 0.2393 | +0.09 | -0.13 |
| **hybrid** (committed, built on `mcs_ours`) | 0.2680 | **+5.80** | **+4.43** | 0.2009 | **+7.65** | **+6.64** | 0.2243 | **+6.36** | **+6.16** |
| P109's combination (committed) | 0.2651 | +6.83 | +5.48 | 0.1996 | +8.25 | +7.25 | 0.2376 | +0.80 | +0.58 |

**The tests** (`tests.csv`; DM statistic, raw p, Holm p within the family):

| Test | 15 min | 30 min | last hour (S1) |
|---|---|---|---|
| H1: the hybrid beats `mcs_cpp` (one-sided) | gain +4.43, stat 6.41, p below 0.001 (Holm below 0.001) | +6.64, stat 8.34, p below 0.001 (Holm below 0.001) | +6.16, stat 3.80, p below 0.001 (Holm below 0.001) |
| H2: `mcs_ours` vs `mcs_cpp` (two-sided; estimate is `mcs_ours`'s gain over `mcs_cpp`) | -1.45, stat -3.03, p 0.002 (Holm 0.005) | -1.09, stat -2.45, p 0.014 (Holm 0.014) | -0.22, stat -0.36, p 0.720 (Holm 0.720) |
| For comparison: the hybrid beats `mcs_ours` (committed, one-sided, raw p; `results/pipeline_nse_v2/tests_<target>.csv`) | +5.80, stat 5.97, p below 0.001 | +7.65, stat 7.94, p below 0.001 | +6.36, stat 3.73, p below 0.001 |

**Delta and the verdict** (`verdict.csv`):

| | 15 min | 30 min |
|---|---|---|
| the hybrid's gain over `mcs_ours` (headline) | 5.80 | 7.65 |
| the hybrid's gain over `mcs_cpp` | 4.43 | 6.64 |
| Delta [95% stationary-bootstrap interval] | **-1.37 [-2.15, -0.58]** | **-1.00 [-1.77, -0.23]** |
| outcome | headline overstated | headline overstated |

* **Verdict: "headline overstated"** at both horizons. H1: "the hybrid beats `mcs_cpp` at every primary
  horizon".
* Both Delta intervals lie wholly below zero and reach beyond the -0.5 margin; the point estimates are 2.7 and 2.0
  times the margin.
* In the last hour the two benchmarks cannot be told apart (-0.22%, p 0.720), and the hybrid's gain is 6.16%
  against `mcs_cpp` and 6.36% against `mcs_ours`.

## 4. Which of the three choices matters (REAL DATA, Nifty)

**The Shapley split** of `mcs_ours`'s extra loss over `mcs_cpp` (`attribution.csv`; points of `mcs_cpp`'s QLIKE;
negative means our choice makes the benchmark worse; the three parts add up to the total):

| Subset | Target | Total | expiry-day curve | smoothing | zero mean |
|---|---|---|---|---|---|
| all days | 15 min | -1.45 | -0.84 | -0.62 | +0.01 |
| all days | 30 min | -1.09 | -0.39 | -0.69 | -0.01 |
| all days | last hour | -0.22 | +0.27 | -0.36 | -0.13 |
| expiry days (169) | 15 min | -3.93 | -3.36 | -0.55 | -0.01 |
| expiry days | 30 min | -2.01 | -1.59 | -0.39 | -0.03 |
| non-expiry days (623) | 15 min | -0.61 | +0.02 | -0.64 | +0.01 |
| non-expiry days | 30 min | -0.78 | +0.02 | -0.79 | +0.00 |
| 09:15 anchor only | 15 min | -2.97 | -8.78 | +5.61 | +0.19 |
| 09:15 anchor only | 30 min | -2.84 | -6.94 | +3.90 | +0.19 |
| all other anchors | 15 min | -1.31 | -0.09 | -1.21 | -0.01 |
| all other anchors | 30 min | -0.91 | +0.28 | -1.16 | -0.03 |

**One switch at a time** (E1 and E2, exploratory, two-sided DM, raw p; `tests.csv`). Change in loss when one
switch of `mcs_ours` is set to the paper's choice (% of `mcs_ours`'s QLIKE; negative: the paper's choice is better),
and gain when one switch of `mcs_cpp` is set to ours (% of `mcs_cpp`'s QLIKE; negative: our choice is worse):

| Switch | E1, from `mcs_ours`: 15 min | 30 min | last hour | E2, from `mcs_cpp`: 15 min | 30 min | last hour |
|---|---|---|---|---|---|---|
| expiry-day curve | -0.69 (p 0.085) | -0.31 (0.356) | +0.24 (0.500) | -0.98 (0.040) | -0.46 (0.236) | +0.31 (0.419) |
| smoothing | -0.47 (0.294) | -0.61 (0.159) | -0.39 (0.497) | -0.76 (0.090) | -0.76 (0.063) | -0.33 (0.552) |
| mean | +0.01 (0.727) | -0.01 (0.687) | -0.13 (0.016) | +0.01 (0.621) | -0.00 (0.967) | -0.13 (0.029) |

* No single switch is clearly significant on its own at 15 or 30 minutes; together they are (H2). The expiry-day
  curve and the smoothing each cost between about 0.3 and 1.0 points, whichever end one starts from.
* **The expiry-day curve hurts on expiry days.** That is where it should help. It is estimated from the expiry days
  of the training window only (197 of the 927 training days in the last refit, fewer in early refits), and its
  extra noise costs more than the real difference in shape gains. At the 09:15 forecast it costs 8.78 points: the
  first-minute spike on expiry days is especially noisy (the raw expiry-day factor at 09:15 is 0.128 against 0.102
  for one curve in the last refit, `posthoc_diurnal_curves.txt`).
* **The smoothing hurts after the open and helps at the open.** Post hoc and descriptive
  (`posthoc_diurnal_curves.txt`, last Nifty refit): the 15:00 minute has a one-minute spike in the diurnal curve
  (raw factor 0.013 against about 0.002 in the minutes around it). The centred 5-minute smoothing spreads it into
  14:58 and 14:59, so the curve's total over 14:45 to 14:59 rises from 0.034 to 0.039 and over 15:00 to 15:14 falls
  from 0.050 to 0.045. At those anchors `mcs_ours` is 5.71% (14:45) and 6.67% (15:00) worse than `mcs_cpp` at 15
  minutes, and `ours_no_expiry` (smoothing kept) is as bad (5.74% and 7.79%; `gain_by_anchor.csv`). At 09:15 the
  smoothing pads the curve with its own first value, which raises the curve's total over 09:15 to 09:29 from 0.191
  to 0.210; this happens to correct part of mcsGARCH's too-low opening forecast (P109 found the 09:15 forecast too
  low by a factor of about 1.17), so it helps there.
* This matches P109's time-of-day factors for `mcs_ours` (0.830 at 14:45 and 1.286 at 15:00, "the 14:45 to 15:00
  window is forecast too high and the 15:00 to 15:15 window too low"), which P109 described as "where, not why".
  The smoothing of our benchmark is a likely "why". This is an interpretation, not a registered test.
* **The ARMA(1,1) mean** changes nothing that matters: at most 0.13 points (last hour), and 0.01 points or less at
  15 and 30 minutes.

## 5. Expiry days, the open and the rest of the day (REAL DATA, Nifty)

**S2, expiry days only** (169 days; Holm over 4 rows):

| Test | 15 min | 30 min |
|---|---|---|
| H1: the hybrid beats `mcs_cpp` | gain +2.97, stat 1.66, p 0.050 (Holm 0.050) | +4.35, stat 2.54, p 0.006 (Holm 0.018) |
| H2: `mcs_ours` vs `mcs_cpp` | -3.93, stat -3.84, p below 0.001 (Holm 0.001) | -2.01, stat -2.37, p 0.019 (Holm 0.038) |

* On expiry days the headline's gain (6.64% and 6.23% against `mcs_ours`) shrinks to 2.97% and 4.35% against
  `mcs_cpp`. At 15 minutes the hybrid's edge on expiry days only just passes (Holm p 0.0498).
* On the 623 non-expiry days (E5, exploratory) the hybrid gains 4.92% and 7.42% over `mcs_cpp` (p below 0.001),
  and `mcs_ours` is 0.61% and 0.78% worse than `mcs_cpp` (p 0.161 and 0.093).

**E4, the 09:15 anchor and the other anchors** (exploratory, raw p):

| Anchors | 15 min: hybrid's gain over `mcs_cpp` | `mcs_ours` vs `mcs_cpp` | 30 min: hybrid's gain over `mcs_cpp` | `mcs_ours` vs `mcs_cpp` |
|---|---|---|---|---|
| 09:15 only | +10.72 (p below 0.001) | -2.97 (p 0.278) | +21.70 (p below 0.001) | -2.84 (p 0.140) |
| all other anchors | +3.83 (p below 0.001) | -1.31 (p 0.003) | +5.11 (p below 0.001) | -0.91 (p 0.040) |

Against `mcs_ours` the hybrid's gains were 13.30 and 23.86 at 09:15 and 5.08 and 5.96 elsewhere (`attribution.csv`).
The largest single changes are at 15:00 at 15 minutes: the hybrid's gain is 12.51% against `mcs_ours` but 6.68%
against `mcs_cpp`; at 14:45 it is 4.81% against `mcs_ours` and -0.62% against `mcs_cpp` (`gain_by_anchor.csv`).
So part of what the hybrid's correction did at 14:45 and 15:00 was to undo our benchmark's smoothing.

**By expiry regime** (descriptive, `attribution.csv`, 15 minutes): `mcs_ours` is worse than `mcs_cpp` in all three
regimes: by 1.57 points with several weekly expiries (312 days), 0.53 with one Thursday expiry (192 days) and 2.03
with one Tuesday expiry (288 days); at 30 minutes by 1.07, 0.14 and 1.91.

## 6. P109's combination against the paper's specification (REAL DATA, Nifty)

**S3** (Holm over 2 rows): the combination beats `mcs_cpp` by **5.48%** at 15 minutes (stat 4.82, Holm p below
0.001) and **7.25%** at 30 minutes (stat 4.78, Holm p below 0.001); against `mcs_ours` it gained 6.83% and 8.25%.
In the last hour it is 0.58% better than `mcs_cpp` (descriptive; P109 found no last-hour gain either). P109's
comparison of the hybrid with the combination does not involve the benchmark and is unchanged.

## 7. Bank Nifty and Fin Nifty (REAL DATA, nothing retuned)

S4 and S5 (Holm over 4 rows each) and the decision rule applied descriptively (`verdict.csv`, `tests.csv`):

| Index, target | hybrid's gain vs `mcs_ours` | vs `mcs_cpp` (H1 Holm p) | `mcs_ours` vs `mcs_cpp` (H2 Holm p) | Delta [95%] | outcome |
|---|---|---|---|---|---|
| Bank Nifty, 15 min (660 days) | 3.92 | 4.73 (below 0.001) | +0.84 (0.003) | +0.81 [0.23, 1.46] | headline understated |
| Bank Nifty, 30 min | 5.03 | 6.20 (below 0.001) | +1.23 (below 0.001) | +1.17 [0.53, 1.85] | headline understated |
| Fin Nifty, 15 min (604 days) | 4.90 | 4.03 (below 0.001) | -0.91 (0.059) | -0.87 [-1.67, -0.06] | inconclusive |
| Fin Nifty, 30 min | 6.09 | 5.18 (below 0.001) | -0.97 (0.074) | -0.91 [-1.92, 0.11] | inconclusive |

Shapley split (points of `mcs_cpp`'s QLIKE; `attribution.csv`):

| Index, target | total | expiry-day curve | smoothing | zero mean |
|---|---|---|---|---|
| Bank Nifty, 15 min | +0.84 | +0.25 | +0.71 | -0.11 |
| Bank Nifty, 30 min | +1.23 | +0.44 | +0.87 | -0.08 |
| Fin Nifty, 15 min | -0.91 | -1.26 | +0.41 | -0.07 |
| Fin Nifty, 30 min | -0.97 | -1.50 | +0.59 | -0.05 |

* **The hybrid beats `mcs_cpp` on every index at both horizons.**
* **Whether our choices help depends on the index.** The smoothing helps on Bank Nifty and Fin Nifty and hurts on
  Nifty; the expiry-day curve helps a little on Bank Nifty and hurts on Nifty and Fin Nifty. The zero mean is
  slightly worse than the ARMA mean on both (by 0.05 to 0.11 points; E1 raw p 0.004 or below at 15 minutes), a
  small but consistent effect.
* **Fin Nifty expiry days** (82 days, descriptive, `losses.csv`): `mcs_cpp` is 4.63% and 5.91% better than
  `mcs_ours`, and the hybrid is 0.77% and 0.70% *worse* than `mcs_cpp` there. On Bank Nifty expiry days (84) the
  hybrid gains 4.21% and 6.44% over `mcs_cpp`.

## 8. The squared mean, the fitted ARMA terms and the fits (REAL DATA)

* **E3, the squared conditional mean in the forecast** (exploratory): leaving it out raises `mcs_cpp`'s loss by
  0.005% at 15 minutes (p 0.005), 0.003% at 30 minutes (p 0.048) and 0.016% in the last hour (p 0.069). The
  registered choice (include it) is slightly better, by amounts far too small to matter.
* **The fitted ARMA(1,1) terms** (`params_summary.csv`, `params_by_refit.csv`, means over refits): Nifty phi -0.47
  (from -0.93 to 0.51 across the 38 refits) and theta 0.48 (from -0.51 to 0.94), so the two nearly cancel (phi +
  theta from -0.003 to 0.020, mean 0.011); Bank Nifty phi 0.58 and theta -0.61 (phi + theta -0.021); Fin Nifty phi
  0.62 and theta -0.64 (-0.017). This is what an ARMA(1,1) does on returns that are nearly white noise: the large
  swings of phi and theta between refits are a flat ridge in the fit, not a change in the market.
* **The intraday GARCH** is almost the same in every model: on Nifty, alpha 0.045 and beta 0.950 for both
  `mcs_ours` and `mcs_cpp` (means over 38 refits), Student t degrees of freedom 6.45 and 6.59.
* **Fits.** All 891 model rows of `params_by_refit.csv` converged (8 distinct fits per refit; `mcs_cpp_varonly`
  repeats `mcs_cpp`'s row).

## 9. What we expected, and what happened (REAL DATA)

| Registered expectation | Outcome |
|---|---|
| H1 holds at both horizons | **as expected** (Holm p below 0.001 at both) |
| `mcs_cpp` forecasts somewhat worse than `mcs_ours` | **not as expected: better**, by 1.45% and 1.09% (Holm p 0.005 and 0.014) |
| mostly because of the expiry-day curve on expiry days | **the expiry-day curve is indeed the largest piece on expiry days, but it works against our version**; the smoothing is an equal piece overall |
| the ARMA mean changes nearly nothing | **as expected** (0.01 points or less at 15 and 30 minutes) |
| overall outcome "headline understated" or "no material difference" | **not as expected: "headline overstated"** on Nifty; "understated" on Bank Nifty; "inconclusive" on Fin Nifty |

## 10. What this means for how the project describes its benchmark

Under the registered rule (plan Section 6, outcome "headline overstated"), the sentence to use is:

> "Our implementation of mcsGARCH (a separate expiry-day diurnal curve, a 5-minute smoothing of the diurnal curve
> and a zero intraday mean) forecasts Nifty slightly worse than mcsGARCH as Chhimwal, Pandey and Pandey specify it
> (one diurnal curve, no smoothing, ARMA(1,1) mean, Student t), by 1.45% and 1.09% of its loss at 15 and 30
> minutes (P116). Against the paper's specification the hybrid's gains are 4.43% and 6.64%, smaller than the 5.80%
> and 7.65% reported against our implementation, and still significant (Holm p below 0.001)."

* **Headline.** The Nifty headline should be restated against `mcs_cpp`: **4.43% at 15 minutes and 6.64% at 30
  minutes** (last hour 6.16%), with 5.80% and 7.65% kept only as the comparison against our earlier version. Where
  a document says "our implementation of mcsGARCH", it should add that this implementation is slightly weaker than
  the paper's specification on Nifty, and name the expiry-day curve and the smoothing.
* **Not "only machine learning".** The hybrid's correction partly undoes our own benchmark's smoothing at 14:45
  and 15:00 (Section 5). This adds to P109's point: part of the reported gain was a gain over a fixable benchmark.
* **Other indices.** Bank Nifty's headline (3.92% and 5.03%) is conservative: against the paper's specification
  the gain is 4.73% and 6.20%. Fin Nifty's (4.90% and 6.09%) cannot be said to change significantly; against the
  paper's specification it is 4.03% and 5.18%. Quoting both benchmarks for these indices avoids picking the one
  that flatters the hybrid.
* **P109's combination** beats the paper's specification by 5.48% and 7.25% (Holm p below 0.001).
* **Other results measured "against mcsGARCH"** (P106's contributions, P109's shares, the `mcsgarch_iv`
  comparisons, the referee checks) use `mcs_ours` as the unit and were not recomputed here. Their comparisons
  between hybrids or between extensions do not depend on the benchmark; statements of the size of a gain over
  mcsGARCH do, and should be read as gains over our implementation.
* **Bank Nifty's or Fin Nifty's documents** should not adopt `mcs_cpp` as a better benchmark on the strength of
  Nifty: which version forecasts better differs by index.

## 11. Checks (REAL DATA unless marked SYNTHETIC)

* **Reproduction of the committed benchmark** (`reproduction_check.csv`). In the registered run, the rebuilt
  `mcs_ours` and the targets y are identical to the last bit to the committed forecasts on every row: Nifty 19,800
  (15 minutes), 19,008 (30 minutes) and 792 (last hour); Bank Nifty 16,500 and 15,840; Fin Nifty 15,100 and 14,496
  (and the last hour of both, 660 and 604, in the phase-1 check). `analyze` refuses to run otherwise.
* **The same days as P109.** P109's committed daily QLIKE of mcsGARCH and of the hybrid equal this study's on every
  day of every index and target used (largest relative difference 0), so P109's committed daily losses of the
  combination sit on the same days and targets.
* **P109 still reproduces with the new code path** (phase 1, plan Section 12): P109's own driver rebuilt all 12
  Nifty variants and its daily losses are byte for byte identical to P109's committed files.
* **Inputs.** Every raw file matches `data/MANIFEST.csv` (`run_info.json`, `input_sha256.<index>.manifest_check`):
  1,891 files for Nifty, 1,891 for Bank Nifty and 1,711 for Fin Nifty, none differing.
* **No look-ahead**, by construction and by test: every model is fitted on the days before its block only; the
  conditional mean of minute i uses returns up to minute i - 1 only (test
  `test_conditional_mean_matches_definition`); the forecast with the mean part equals a Monte Carlo of the full
  model (`test_full_forecast_matches_monte_carlo`, SYNTHETIC); the driver's `mcs_ours` and `mcs_cpp` equal direct
  fits of `McsGarch` bit for bit (`test_driver_equals_direct_fits`, SYNTHETIC).
* **Tests.** `tests/test_p116_benchmark_as_printed.py` (SYNTHETIC only, 10 tests, committed with the plan) passes.
  The full suite result is in Section 14.

## 12. Run time

* **Forecasts** (stage 1, 2 worker processes), on 3 November 2026 (`run_stdout_*.txt`, `run_info.json`,
  `stage1`): Nifty 3.0 minutes (10:58 to 11:01 UTC, including 62 s to load the market data; 38 refits, 8 distinct
  fits each); Bank Nifty 2.4 minutes and Fin Nifty 2.2 minutes (finished 11:05:40 UTC). The jobs' own times add up to
  226.1, 175.2 and 151.1 seconds (`timing.csv`).
* **Analysis** (stage 2): 8.4 seconds (`run_info.json`, `seconds_stage2`).
* The plan expected 10 to 20 minutes for Nifty and 5 to 10 for each other index. The whole registered run took
  about 8 minutes (10:58:00 to 11:05:52 UTC).

## 13. Deviations from the plan

1. **None in the design.** Every registered model, test, family and output was run as written, on all three
   indices, with no change to the code after the plan commit (`analysis_commit` in `run_info.json` is the plan
   commit). Nothing was dropped; no run was restarted; no block was recomputed.
2. **One post hoc, descriptive addition** after the registered run had been seen: `experiments/p116_posthoc_diurnal_curves.py`
   prints the diurnal curves of the last Nifty refit (raw and smoothed, one curve and the expiry-day curve) at the
   open, around 15:00 and at the close (`posthoc_diurnal_curves.txt`). It was added to explain where the smoothing
   and the expiry-day curve change the forecasts (Section 4). It computes no forecast, loss or test and changes no
   registered number.
3. **Commands.** The plan's commands set `PYTHONPATH=src`; the run used the absolute path of the same `src` folder
   (a sandbox rule), with `--plan-commit 19c5486cd39e9944764ac9a1ea7c3ab3bbb7b63b`. Same code.
4. **Reading notes, not deviations.** `losses.csv` gives each model's gain over each benchmark in % of that
   benchmark's QLIKE (the project's "points") and the raw QLIKE difference (`loss_diff_vs_*`). The S2 expiry-day H1
   row at 15 minutes has Holm p 0.0498, just below 0.05; it is reported as it falls. The decision rule applied to
   Bank Nifty and Fin Nifty is descriptive, as registered.

## 14. Caveats

1. **"As the paper specifies" is as close as the code allows.** The session grid (375 minutes, no overnight gap in
   the first return, needed for the targets), the two-step ARMA estimation, the constant in the daily AR(1) and the
   cross-day recursion are our choices (plan Section 1). The paper's software settings are not published.
2. **The hybrid is built on `mcs_ours`.** Its gain over `mcs_cpp` compares two fixed sets of forecasts. A hybrid
   built on `mcs_cpp` might gain a different amount over `mcs_cpp`; that would be a new model and was not fitted.
3. **One choice per index would be specification search.** On Nifty the paper's specification is better, on Bank
   Nifty ours is. Picking the better version for each index after seeing these results would flatter the
   benchmark, and picking the worse would flatter the hybrid; neither should be done. The registered comparison is
   against the paper's specification.
4. **Same days.** The 792 Nifty test days are the ones on which the headline was found; this study changes the
   benchmark, not the data.
5. **The hybrid is one seed set.** P99's seed-to-seed range of the hybrid's gain (0.28 and 0.35 points) is well
   below the change found here (1.37 and 1.00 points).
6. **The single-switch rows are exploratory.** The split among the expiry-day curve and the smoothing comes from
   point estimates (Shapley) and from unadjusted tests, none of which is clearly significant alone at 15 or 30
   minutes. The explanation of the smoothing effect at 14:45 and 15:00 is a post hoc reading of the diurnal curve.
7. **The ARMA(1,1) fit is on a flat ridge** (phi and theta nearly cancel and swing between refits). Its forecasts
   are stable because only phi + theta and the small constant matter, but its individual coefficients should not
   be interpreted.

**Full test suite.** Run after the analysis and before the result commit, on 3 November 2026 (finished 11:19:53
UTC, about 5 minutes, with the raw data present): 668 tests collected in 48 files, 668 passed, 0 failed, 0 skipped.
The only warnings are three existing scipy `RuntimeWarning`s in `tests/test_sensex_confound.py`. A first attempt
(run in the background, 11:10:00 to 11:13:53 UTC) left an empty log, so its result is unknown and it is not
counted; the suite was run again in the foreground with its full output saved, which is the result given here.

## 15. Files

All in `results/p116_benchmark_as_printed/` (REAL DATA):

* `tests.csv`: every p-value (77 rows), with the registered columns; `tests_detail.csv` adds identifiers and
  samples.
* `verdict.csv`: the registered verdict for Nifty (and the rule applied descriptively to Bank Nifty and Fin Nifty),
  with the gains, Delta and its interval.
* `losses.csv`: every model's mean daily QLIKE by index, target and subset (all days, expiry and non-expiry days,
  regimes, the 09:15 anchor and the others), with its gain over `mcs_ours` and over `mcs_cpp` in % and as a raw
  QLIKE difference.
* `attribution.csv`: the Shapley split and the one-switch changes, by index, target and subset.
* `gain_by_anchor.csv`: the gains at every anchor minute.
* `params_by_refit.csv`, `params_summary.csv`: every refit's ARMA and GARCH parameters, and their summaries.
* `daily_losses_<index>_<target>.csv`: each model's daily QLIKE (with the combination's), from which every number
  can be recomputed.
* `reproduction_check.csv`, `timing.csv`, `run_info.json`: checks, timings, plan commit, settings, input hashes and
  versions.
* `run_stdout_nifty.txt`, `run_stdout_banknifty.txt`, `run_stdout_finnifty.txt`, `analyze_stdout.txt`: logs.
* `posthoc_diurnal_curves.txt`: the post hoc descriptive diurnal curves (Section 13).

The forecasts of every model are in `data/processed/p116/` (git-ignored).

## 16. How to reproduce

From the repository root, with the raw data listed in `data/MANIFEST.csv` in `data/raw`:

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py run --index NIFTY --workers 2 \
    > results/p116_benchmark_as_printed/run_stdout_nifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py run --index BANKNIFTY --workers 2 \
    > results/p116_benchmark_as_printed/run_stdout_banknifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py run --index FINNIFTY --workers 2 \
    > results/p116_benchmark_as_printed/run_stdout_finnifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py analyze \
    --plan-commit 19c5486cd39e9944764ac9a1ea7c3ab3bbb7b63b > results/p116_benchmark_as_printed/analyze_stdout.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_posthoc_diurnal_curves.py \
    > results/p116_benchmark_as_printed/posthoc_diurnal_curves.txt
```

About 8 minutes on 2 cores. The synthetic tests: `PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q
tests/test_p116_benchmark_as_printed.py`. A synthetic end-to-end run of all stages: add `--synthetic` (labelled
SYNTHETIC).
