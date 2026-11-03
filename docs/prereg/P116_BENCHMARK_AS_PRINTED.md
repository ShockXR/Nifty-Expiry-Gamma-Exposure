# P116 pre-registration: how large is the hybrid's gain over mcsGARCH as the paper specifies it?

**Status.** Written on 3 November 2026 (phase 1) by an AI agent (Claude) working for Shreyan Thanki. **This design
is committed to git before any forecast of a new benchmark variant has been made on real data.** No forecast of
`mcs_cpp` or of any other variant of this study has been made on real data, and no loss, gain, test statistic or
bootstrap draw of this study has been computed on real data. The only real-data computations so far are the
reproduction checks in Section 12: this study's driver rebuilt the committed benchmark (`mcs_ours`) on Nifty, Bank
Nifty and Fin Nifty and compared its forecasts and targets with the committed ones, number by number, without
computing any loss; and P109's own driver, with this study's small change to it, rebuilt P109's committed Nifty
forecasts, whose daily losses were compared with P109's committed daily losses (those losses are already public in
`results/p109_mcsgarch_extensions/`). The script and its tests were also run end to end on a simulated market
(SYNTHETIC). The commit that adds this file is the time stamp; phase 2 (the run) starts only after that commit
exists and has been pushed. The code that phase 2 runs (`experiments/p116_benchmark_as_printed.py`, one new code
path in `experiments/p109_mcsgarch_extensions.py`, new functions at the end of `src/niftygex/volatility.py`) and its
tests (`tests/test_p116_benchmark_as_printed.py`, synthetic data only) are in the same commit. Phase 2 writes
`results/p116_benchmark_as_printed/` and `docs/P116_BENCHMARK_AS_PRINTED.md`. All results will be REAL DATA.

## In plain words

The project's forecasting contest uses mcsGARCH, the anchor paper's model, as the benchmark: our hybrid
(mcsGARCH times a correction learnt by LightGBM) is 5.80% and 7.65% better than it at 15 and 30 minutes (QLIKE,
792 Nifty test days). An audit found that the benchmark in the contest is not exactly the paper's model. Our
version makes three choices the paper does not make:

1. **A second time-of-day curve for expiry days.** mcsGARCH multiplies a daily volatility level by a time-of-day
   curve (how much the market usually moves at 09:15, at 09:16, and so on). The paper estimates one curve for all
   days. Our version estimates one curve for ordinary days and a separate one for expiry days.
2. **Smoothing of the curve.** The paper takes the plain average for each minute. Our version smooths each curve
   with a centred 5-minute moving average.
3. **The intraday mean.** The paper models the expected one-minute return with an ARMA(1,1) (a small amount of
   memory in returns). Our version sets the expected return to zero.

The professor's obvious question is: **how big is the hybrid's gain over mcsGARCH as the paper specifies it, and
do our three choices make the benchmark better or worse?** This study answers it. We refit mcsGARCH with the
paper's three choices (`mcs_cpp`), with exactly the same walk-forward, days and targets as the contest, and
compare: the hybrid against `mcs_cpp` (is the gain still there, and how big?), and our version against `mcs_cpp`
(does our extension make the benchmark stronger or weaker?). We also refit every mix of the three choices, so we
can say which one matters.

## Why this study

* `src/niftygex/forecast.py` (`WalkForwardConfig`, lines 76 and 77: `expiry_diurnal = True`, `diurnal_smooth =
  5`; line 293: `McsGarch(expiry_diurnal=cfg.expiry_diurnal, smooth=cfg.diurnal_smooth)`) builds the contest's
  benchmark with the default zero mean (`McsGarch.mean = "zero"`, `src/niftygex/volatility.py`).
* The paper's equation 7 has one diurnal curve, s_i = (1/T) sum_t eps_{t,i}^2 / sigma_t^2, with no smoothing, and
  its Table 2 is titled "ARMA(1,1)-mcsGARCH(1,1)" with Student-t errors (CPP, Section 3.1, pages 6 and 7). The
  project's replication of Table 2 fits `McsGarch(mean="arma11", dist="t", smooth=1)` with one curve
  (`experiments/replicate_paper.py`, line 497).
* The project's documents now say "our implementation of mcsGARCH" (`ITERATION_LOG.md`, the audit entry). That
  is accurate but leaves open whether the headline would be smaller or larger against the paper's own
  specification. A referee or co-author will ask; the answer should come from a registered test, not from a
  choice made after seeing which benchmark flatters the hybrid.

## Prior knowledge (what was known when this was written)

All REAL DATA, from committed files.

* **The headline** (`results/pipeline_nse_v2/scoreboard_<target>.csv`, 792 test days, 10 August 2023 to
  2 November 2026): mean daily QLIKE of `mcs_ours` 0.2845 (15 minutes), 0.2176 (30 minutes), 0.2395 (last hour);
  of the hybrid 0.2680, 0.2009, 0.2243; the hybrid's gain 5.80, 7.65 and 6.36% of `mcs_ours`'s QLIKE. On the 169
  expiry days: `mcs_ours` 0.3454 and 0.2591, the hybrid's gain 6.64% and 6.23%.
* **Bank Nifty** (`results/pipeline_nse_banknifty/`, 660 test days to 22 April 2026): the hybrid's gain 3.92% and
  5.03%; **Fin Nifty** (`results/pipeline_nse_finnifty/`, 604 test days to 27 January 2026): 4.90% and 6.09%.
* **How much the headline moves with LightGBM's seeds alone** (P99, `results/seed_bootstrap/summary.csv`, 20
  further seed sets): the hybrid's gain over `mcs_ours` ranges from 5.64% to 5.92% (standard deviation 0.075) at
  15 minutes and from 7.76% to 8.11% (0.112) at 30 minutes; in the last hour SD 0.244.
* **P109** (`docs/P109_MCSGARCH_EXTENSIONS.md`): five fixes to `mcs_ours` combined (the "combination") are 6.83%
  and 8.25% better than `mcs_ours`, and the hybrid does not beat the combination. The combination keeps
  `mcs_ours`'s expiry curve, smoothing and zero mean.
* **The paper's model has been fitted on real data only in-sample**, in the replication of Table 2
  (`results/replication_nse/table2_mcsgarch.csv`, 64-day windows, a 360-minute session with the overnight gap in
  the first return). There the ARMA(1,1) coefficients nearly cancel: for example Nifty on the stated before window
  phi -0.331 and theta 0.345 (phi + theta 0.014), Bank Nifty phi 0.739 and theta -0.751 (phi + theta -0.012). The
  project's iteration log (entry on the diurnal smoothing) reads this as what ARMA(1,1) does on near-white-noise
  data. So we expect the mean to change the forecasts very little.
* **No forecast of mcsGARCH with one diurnal curve, without smoothing or with an ARMA mean has been made in the
  forecasting walk-forward on real data.** The repository was searched for `expiry_diurnal` and `smooth` (code,
  results and documents); every forecasting run uses `McsGarch(expiry_diurnal=True, smooth=5)` with a zero mean
  (`scripts/run_pipeline.py`, `experiments/referee_forecast.py`, P106 and P109).

## 1. The models

| Name | Expiry-day curve | Smoothing | Intraday mean | Role |
|---|---|---|---|---|
| `mcs_ours` | separate curve for expiry days | centred 5-minute moving average | zero | the committed benchmark, rebuilt and checked to equal the committed forecasts exactly |
| **`mcs_cpp`** | **one curve for all days** | **none** | **ARMA(1,1)** | **mcsGARCH as the paper specifies it** |
| `ours_no_expiry` | one curve | 5-minute | zero | one switch of `mcs_ours` set to the paper's choice |
| `ours_no_smooth` | separate | none | zero | one switch of `mcs_ours` set to the paper's choice |
| `ours_arma` | separate | 5-minute | ARMA(1,1) | one switch of `mcs_ours` set to the paper's choice |
| `cpp_expiry` | separate | none | ARMA(1,1) | one switch of `mcs_cpp` set to ours |
| `cpp_smooth` | one curve | 5-minute | ARMA(1,1) | one switch of `mcs_cpp` set to ours |
| `cpp_zero_mean` | one curve | none | zero | one switch of `mcs_cpp` set to ours |
| `mcs_cpp_varonly` | one curve | none | ARMA(1,1) | the `mcs_cpp` fit, forecast without the squared conditional mean (exploratory) |
| `hybrid` | | | | the committed `hybrid_base` forecasts (seed set 7), unchanged; it is built on `mcs_ours` |
| `combination` | | | | P109's combination, from its committed daily losses (`results/p109_mcsgarch_extensions/daily_losses_<index>_<target>.csv`), unchanged |

The eight models from `mcs_ours` to `cpp_zero_mean` are all 2 x 2 x 2 combinations of the three switches, so the
difference between `mcs_ours` and `mcs_cpp` can be split among the switches exactly (Section 3).

**Everything else is the same in every model**, and as committed: the daily level h_t from the AR(1)-EGARCH(1,1)
with Student-t errors (`daily_egarch_variance(daily returns, train_size=start)`, with its fallbacks); the
intraday GARCH(1,1) on the normalised residual x = eps / sqrt(h_t s_i), Student-t, fitted by maximum likelihood
(`fit_garch11`); the recursion running across day boundaries; the filter started at the unconditional level; the
forecast cut at the close.

**`mcs_cpp` in detail** (`McsGarch(mean="arma11", dist="t", expiry_diurnal=False, smooth=1)`, the replication's
model):

1. The ARMA(1,1) mean r_n = c + phi r_{n-1} + theta e_{n-1} + e_n is fitted by conditional least squares
   (`fit_arma11`) on the one-minute returns of all training days, run as one series across day boundaries.
2. The residuals eps = e give the diurnal curve s_i, the plain average of eps^2 / h_t over the training days for
   each minute (the paper's equation 7), one curve, no smoothing.
3. The intraday Student-t GARCH(1,1) is fitted to x = eps / sqrt(h_t s_i).
4. All three steps are refitted at every refit, on the training days only, and the model is then filtered over all
   days, exactly as `mcs_ours`.

**The forecast of the target when the mean is ARMA(1,1).** The target is the sum of squared one-minute returns
over the window, y = sum_{j=i}^{i+H-1} r_j^2. With r_j = mu_j + eps_j, where mu_j is the conditional mean and
eps_j has mean zero given the past, E[r_j^2] = E[mu_j^2] + E[eps_j^2]. The forecast made at the end of minute i - 1
is therefore

    F = sum_j h_t s_j E[q_j]  +  sum_j (m_j^2 + V_j),

where the first sum is exactly `McsGarch.forecast_variance` (the expected path E[q_j] = qbar + (alpha + beta)^k
(q_i - qbar), k = j - i, as for `mcs_ours`), and the second is the expected squared conditional mean
(`arma11_mean_square_forecast`): m_i = mu_i is known (c + phi r_{i-1} + theta e_{i-1}) and V_i = 0; for later
minutes mu_j = c + phi mu_{j-1} + (phi + theta) e_{j-1}, so m_j = c + phi m_{j-1} and V_j = phi^2 V_{j-1} + (phi +
theta)^2 h_t s_{j-1} E[q_{j-1}]. **We include the squared conditional mean.** Reason: the forecast that minimises
expected QLIKE is the conditional expectation of the target (Patton 2011), and under the paper's model the
conditional expectation of a sum of squared returns includes the squared mean; leaving it out would forecast the
variance of the residuals, a different quantity from the target. Both forecasts come from the same fit;
`mcs_cpp_varonly`, without the term, is reported as an exploratory check (E3) of how much the choice matters. We
expect it to matter very little (Prior knowledge). For a zero mean the term is 0, so `mcs_ours`'s forecasts do not
change.

**What the paper leaves unspecified, and our choice** (each the same for every model of this study):

| Point | The paper | Choice here | Reason |
|---|---|---|---|
| Session minutes and the first return of the day | not stated; its Table 1 count (64 x 360 - 1 = 23,039) implies a 360-minute session with the overnight gap in the first return, which the replication follows (09:15 to 15:14) | the forecasting grid: 375 minutes, 09:15 to 15:29, the first return from the first bar's own open (no overnight gap); missing returns set to 0, as in `build_market_from_nse` | the targets, the hybrid and the committed benchmark are defined on this grid; the last-hour target needs 15:15 to 15:29; an overnight gap in the 09:15 return would make the model forecast a quantity the target does not contain |
| How s_i is estimated | equation 7: the mean of eps^2 / sigma_t^2 over the T days of the sample | the mean over all training days of each refit (the replication's estimator, `McsGarch._pattern`, with `smooth=1`) | the paper fits one sample and makes no forecast; a walk-forward must use only earlier days |
| How the ARMA(1,1) is estimated | not stated (the R package rugarch, the usual software, estimates mean and variance jointly) | two steps, as in the replication: conditional least squares for the mean, then the variance model on its residuals | the replication's code; a joint likelihood is not implemented |
| Whether the intraday mean and GARCH run across days | not stated | across days, as rugarch and the committed code | the committed benchmark does the same |
| The daily AR(1) mean | equation 5 has no constant (u_t = phi r_{t-1}) | a constant is included (the arch package's AR mean), with the committed EWMA fallback rule | the brief fixes "the same daily level"; the same in every model, so it cannot move the comparison |
| Forecasting | the paper makes no forecast | the model's conditional expectation of the target (above), the window cut at the close | |
| Scale of the components | equation 3 is written in volatilities, equation 7 in variances | variances throughout, as in `niftygex.volatility` | Engle and Sokalska's own definition |

## 2. Design

**What is refitted.** Exactly the committed walk-forward of mcsGARCH, through P109's driver (`run_block` in
`experiments/p109_mcsgarch_extensions.py`, which already rebuilds the committed benchmark exactly): the first 150
days are training only; before every block of 21 trading days every model is refitted on all earlier days (days 0
to start - 1), filtered over all days, and forecasts the block. The forecast rows are those of
`features.build_dataset` (every 15 minutes from 09:15; windows of 15 and 30 minutes; the last hour trained on every
60-minute window and evaluated at 14:30) and the target y is computed as there. One job per refit block, 2 worker
processes. The eight distinct fits per refit (`mcs_cpp_varonly` shares `mcs_cpp`'s fit) are listed in Section 1.

**The hybrid and the combination** are not refitted: the committed `hybrid_base` forecasts (headline seed set 7)
and P109's committed daily losses of the combination are used as they are. The hybrid is built on `mcs_ours`; it
is not rebuilt on `mcs_cpp` (that would be a different model; see Section 10).

**Days.** Nifty: 942 days (2 January 2023 to 2 November 2026), 38 refits, **792 test days** (10 August 2023 to
2 November 2026; 19,800 forecasts at 15 minutes, 19,008 at 30 minutes, 792 in the last hour). Bank Nifty: 810 days
to 22 April 2026, 660 test days. Fin Nifty: 754 days to 27 January 2026, 604 test days. The script refuses to run
if the settings, the number of days or the last day differ from the committed run of that index (P109's check).

| Index | Targets | Models refitted |
|---|---|---|
| Nifty | next 15, next 30, last hour | all nine forecast columns of Section 1 (eight fits) |
| Bank Nifty, Fin Nifty | next 15, next 30 (the last hour is written, not analysed) | the same |

## 3. Statistics

All losses are QLIKE, y/f - log(y/f) - 1, averaged within each day first (`niftygex.evaluation.daily_losses`);
every day has the same number of forecasts (checked). L_m is model m's mean daily QLIKE.

* **The hybrid's gain over a benchmark b**: G_b = 100 x (L_b - L_hybrid) / L_b, in % of the benchmark's QLIKE.
  G_ours is the committed headline (5.80 and 7.65); G_cpp is the gain over the paper's specification.
* **The change in the headline**: Delta = G_cpp - G_ours, in percentage points. Delta has the sign of L_cpp -
  L_ours: positive means `mcs_cpp` forecasts worse than `mcs_ours`, so the headline (against `mcs_ours`)
  understates the gain over the paper's specification. Reported with a 95% stationary-bootstrap percentile interval
  (days resampled jointly for the three models in blocks of mean length 10, 10,000 draws, seed 116;
  `stationary_bootstrap_indices` and `bootstrap_means` from `experiments/referee_forecast.py`).
* **Tests**: `diebold_mariano` from `niftygex.evaluation`, the project's standard (Newey-West variance,
  Harvey-Leybourne-Newbold factor, Student t with n - 1 degrees of freedom), exactly as P109; one-sided or
  two-sided as stated per test.
* **Attribution** (descriptive): the change L_cpp - L_ours is split among the three switches by the Shapley value
  over the 2 x 2 x 2 factorial (each switch's marginal effect averaged over the 6 orders in which the three can be
  switched from the paper's choice to ours); the three parts add up exactly to the change. Also reported: each
  one-switch change from `mcs_ours` and from `mcs_cpp`, in points of `mcs_cpp`'s QLIKE.
* **Descriptive tables**: every model's mean QLIKE and its gain over `mcs_ours` and over `mcs_cpp`, on all days,
  expiry days, non-expiry days, the three expiry regimes, the 09:15 anchor and the other anchors; the gain at every
  anchor minute; every refit's parameters (the ARMA coefficients c, phi, theta; omega, alpha, beta, nu).

## 4. Hypotheses and tests

**Primary family (4 tests, Holm over all four)**, Nifty, all 792 test days:

* **H1, `P_next15_hybrid_vs_cpp` and `P_next30_hybrid_vs_cpp`**: the hybrid has lower mean QLIKE than `mcs_cpp`.
  One-sided DM. Estimate: G_cpp, the hybrid's gain in % of `mcs_cpp`'s QLIKE.
* **H2, `P_next15_ours_vs_cpp` and `P_next30_ours_vs_cpp`**: `mcs_ours` and `mcs_cpp` have different mean QLIKE.
  Two-sided DM on d = loss of `mcs_cpp` minus loss of `mcs_ours` (a positive statistic means our extension makes
  the benchmark better). Estimate: 100 x (L_cpp - L_ours) / L_cpp. Reported with Delta and its interval.

**Secondary families** (each with its own Holm adjustment; they cannot change the primary verdict):

* **S1, last hour (2 tests)**: H1 (one-sided) and H2 (two-sided) for the 14:30 forecast of the last hour
  (`S1_last_hour_hybrid_vs_cpp`, `S1_last_hour_ours_vs_cpp`).
* **S2, expiry days only (4 tests)**: H1 and H2 at 15 and 30 minutes on the 169 expiry days, where a separate
  expiry-day curve should matter most (`S2_<target>_expiry_hybrid_vs_cpp`, `S2_<target>_expiry_ours_vs_cpp`).
* **S3, P109's combination against `mcs_cpp` (2 tests)**: the combination has lower mean QLIKE than `mcs_cpp`, at
  15 and 30 minutes, one-sided (`S3_<target>_combination_vs_cpp`).
* **S4, Bank Nifty (4 tests)** and **S5, Fin Nifty (4 tests)**, nothing retuned: H1 and H2 at 15 and 30 minutes
  (`S4_<target>_hybrid_vs_cpp`, `S4_<target>_ours_vs_cpp`, and `S5_...`). The committed forecasts and the raw data
  of both indices exist (P91, P104, P106 and P109 used them).

**Exploratory** (no adjustment; raw p, labelled exploratory):

* **E1, one switch from `mcs_ours`** (21 rows): `ours_no_expiry`, `ours_no_smooth`, `ours_arma` against
  `mcs_ours`, two-sided, on Nifty (three targets) and Bank Nifty and Fin Nifty (two targets). Estimate: the change
  in loss when that one switch is set to the paper's choice, % of `mcs_ours`'s QLIKE.
* **E2, one switch from `mcs_cpp`** (21 rows): `cpp_expiry`, `cpp_smooth`, `cpp_zero_mean` against `mcs_cpp`,
  two-sided, same samples. Estimate: the gain from setting that one switch to our choice, % of `mcs_cpp`'s QLIKE.
* **E3, the squared conditional mean** (3 rows): `mcs_cpp_varonly` against `mcs_cpp`, two-sided, Nifty, three
  targets.
* **E4, the open and the rest of the day** (8 rows): H1 and H2 at the 09:15 anchor only and at the other anchors,
  15 and 30 minutes (the smoothing pads the curve at 09:15 with its own edge value, so the open is where it could
  matter most).
* **E5, non-expiry days** (4 rows): H1 and H2 at 15 and 30 minutes on the 623 non-expiry days.

Every p-value goes in `results/p116_benchmark_as_printed/tests.csv` with the columns of P106 and P109 (test_id,
hypothesis, role, sample, estimate, statistic, p_raw, sidedness, p_method, p_adjusted, adjust_method), so that the
project's test ledger can read it. Any test added after this commit is labelled "post hoc".

## 5. Decision rule (fixed now; applied by `p116_verdict` in the script)

**At each primary horizon** (15 and 30 minutes), with H2's Holm-adjusted p and Delta:

* **"headline understated"**: H2's Holm p is below 0.05 and `mcs_cpp` has the higher loss (Delta > 0). Our
  extension makes the benchmark better, so the gain over the paper's specification is larger than reported.
* **"headline overstated"**: H2's Holm p is below 0.05 and `mcs_cpp` has the lower loss (Delta < 0).
* In both cases, if |Delta| is at most 0.5 points, the outcome carries the note "by less than the 0.5-point
  margin: a clear but immaterial difference".
* **"no material difference"**: H2's Holm p is 0.05 or more and |Delta| is at most 0.5 points. It carries one of
  two notes: "the 95% interval of the change lies inside +-0.5 points" (the data show the difference is small) or
  "point estimate; the interval reaches beyond +-0.5 points" (the data cannot rule out a material difference).
* **"inconclusive"**: H2's Holm p is 0.05 or more and |Delta| is above 0.5 points.

**Overall**: if both horizons give the same outcome (ignoring the notes), that is the verdict; otherwise the
verdict is "mixed", naming each horizon's outcome. **Separately**, H1 is read as: "the hybrid beats `mcs_cpp`" at
a horizon if its Holm p is below 0.05 there.

**Why 0.5 points.** (1) It is larger than the whole range of the hybrid's gain over the 20 extra seed sets of P99
(0.28 points at 15 minutes and 0.35 at 30 minutes, Prior knowledge): a change of the benchmark within 0.5 points
moves the headline by about as much as rerunning LightGBM with other seeds. (2) It is under a tenth of the
headline gain (5.80 and 7.65 points), so the headline's wording ("about 6% and 8%") would not change. (3) It is a
round number fixed before any result; a smaller margin would turn differences no reader would act on into
"material" ones, a larger one would hide a change of more than a tenth of the gain.

**What we expect** (stated in advance; it changes no rule): H1 holds at both horizons; `mcs_cpp` forecasts
somewhat worse than `mcs_ours`, mostly because of the expiry-day curve on expiry days, with the smoothing a smaller
part and the ARMA mean nearly nothing (the coefficients nearly cancel); the overall outcome is "headline
understated" or "no material difference". This is a guess, not a rule.

## 6. What the documents will say under each outcome

The lead updates the project's summary documents; this study's write-up states the sentence to use.

* **H1 holds at both horizons** (expected): "The hybrid also beats mcsGARCH as the paper specifies it (one diurnal
  curve, no smoothing, ARMA(1,1) mean), by G_cpp% and G_cpp% at 15 and 30 minutes (P116)." If H1 fails at a
  horizon, every place that quotes the headline must add that the hybrid does not beat the paper's specification
  at that horizon, and the headline is reported against both benchmarks.
* **"headline understated"**: "Our implementation of mcsGARCH (with a separate expiry-day curve, a smoothed
  diurnal curve and a zero mean) forecasts better than the paper's specification, by X% of its loss (P116). The
  headline gains are measured against our stronger version, so they are conservative; against the paper's own
  specification they are G_cpp% and G_cpp%." The headline stays against `mcs_ours`.
* **"headline overstated"**: "Our implementation of mcsGARCH forecasts worse than the paper's own specification
  (P116). Against the paper's specification the hybrid's gains are G_cpp% and G_cpp%, smaller than the 5.80% and
  7.65% reported so far." The headline is then restated against `mcs_cpp` everywhere, with the old figures kept
  only as a comparison, and the extension is named as a weakness of the earlier benchmark.
* **"no material difference"**: "Our implementation of mcsGARCH differs from the paper's in three stated ways
  (expiry-day curve, smoothing, zero mean); refitting it exactly as the paper specifies changes the hybrid's gain
  by less than 0.5 points (P116)." With the interval note when the interval reaches beyond 0.5 points.
* **"inconclusive"** or **"mixed"**: both gains are quoted side by side with Delta and its interval, and the
  documents say the data cannot tell whether the extension matters (or at which horizon it does).
* **The attribution** names the switch or switches that carry the change, from the Shapley table, in every case.

## 7. The exact commands

From the repository root, at the commit that adds this file, with the raw data listed in `data/MANIFEST.csv` in
`data/raw`:

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py run --index NIFTY --workers 2 \
    > results/p116_benchmark_as_printed/run_stdout_nifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py run --index BANKNIFTY --workers 2 \
    > results/p116_benchmark_as_printed/run_stdout_banknifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py run --index FINNIFTY --workers 2 \
    > results/p116_benchmark_as_printed/run_stdout_finnifty.txt 2>&1
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py analyze --plan-commit <this commit> \
    > results/p116_benchmark_as_printed/analyze_stdout.txt 2>&1
```

`run` writes one file of forecasts and one of parameters per refit block to `data/processed/p116/` (git-ignored; a
finished block is not recomputed if the run must be restarted) and the SHA-256 of every raw input. `analyze` writes
`tests.csv`, `tests_detail.csv`, `verdict.csv`, `losses.csv`, `attribution.csv`, `gain_by_anchor.csv`,
`reproduction_check.csv`, `params_by_refit.csv`, `params_summary.csv`, `daily_losses_<index>_<target>.csv` (every
model's daily QLIKE, from which every number can be recomputed), `timing.csv` and `run_info.json` (plan commit,
specifications, input hashes, run time, versions).

## 8. Checks inside the run

* The rebuilt `mcs_ours` and y must equal the committed forecasts exactly, on every row of every target and index;
  `analyze` refuses to run otherwise (unless told to report a failure, which would be a deviation).
* P109's committed daily QLIKE of mcsGARCH and of the hybrid must equal this study's (`reproduction_check.csv`),
  so that the combination's committed daily losses sit on the same days and the same targets.
* Every day has the same number of forecasts.

## 9. What will be reported, whatever the result, and the deviations policy

In `docs/P116_BENCHMARK_AS_PRINTED.md`, for a beginner reader, every number labelled REAL DATA and taken from a
file in `results/p116_benchmark_as_printed/`: the plain-language answer first; the verdict under the rule; the main
table (QLIKE of every model at 15 and 30 minutes and in the last hour, the hybrid's gain over each benchmark with
DM p-values); the attribution to the three switches; the expiry-day, last-hour, combination and other-index
results; the exploratory rows with their labels; the fitted ARMA coefficients; what the result means for how the
project describes its benchmark; the reproduction checks; the run time; a "Deviations from the plan" section;
caveats; files; and how to reproduce.

* After this file is committed the design does not change. Any change is a deviation, listed with its reason and
  with whether any outcome had been seen when it was decided. No specification search: no further variant of the
  benchmark is fitted in this study.
* A crash is fixed with the smallest change that lets the run finish, and only if it changes no number of a block
  that already finished.
* If a fit fails to converge, that is reported (`converged` in `params_by_refit.csv`) and its forecasts are used as
  they are.
* If Bank Nifty or Fin Nifty cannot be finished by 17:00 UTC on 3 November 2026 they are dropped and that is
  reported.

## 10. Known weaknesses

1. **"As the paper specifies" is as close as the code allows, not exact.** The session grid, the two-step ARMA
   estimation, the constant in the daily AR(1) and the cross-day recursion are our choices (Section 1). The
   paper's own software settings are not published.
2. **The hybrid is built on `mcs_ours`.** Its gain over `mcs_cpp` compares two fixed sets of forecasts; it does
   not say what a hybrid built on `mcs_cpp` would gain. Rebuilding the hybrid on `mcs_cpp` would be a new model
   and is not part of this study.
3. **Same days.** The 792 test days are the ones on which the headline was found; this study changes the
   benchmark, not the data.
4. **The hybrid is one seed set**; P99 shows its gain moves by up to 0.35 points with the seeds. The benchmarks
   are deterministic.
5. **The EWMA fallback** of the daily EGARCH (two early refits, 42 Nifty test days) is in every model, so it
   cannot move the comparison between them.
6. **Interactions.** The three switches can interact (for example the expiry curve is noisier without smoothing);
   the Shapley split shares interactions equally between the switches involved, which is a convention.

## 11. Code

* `src/niftygex/volatility.py`, new code after all existing code, nothing above changed:
  `arma11_conditional_mean` (mu[t, i] of an ARMA(1,1) mean, from data up to minute i - 1) and
  `arma11_mean_square_forecast` (the expected squared conditional mean over a window; 0 for a zero mean).
* `experiments/p109_mcsgarch_extensions.py`, one new code path: a variant's spec may set `expiry_diurnal`,
  `smooth`, `mean` and `mean_sq`; the defaults (`SPEC_DEFAULTS`: True, 5, "zero", False) are the committed values,
  no P109 variant sets them, so P109 runs exactly as before (Section 12 checks this on real data). The stage-1 log
  line names the study.
* `experiments/p116_benchmark_as_printed.py`, new: `run` (P109's `stage_run` with this study's variants), `check`
  (P109's `stage_check`: the rebuilt `mcs_ours` and y against the committed forecasts, no loss), `analyze`
  (Sections 3 to 5). `--synthetic` runs all stages on a simulated market (labelled SYNTHETIC).
* `tests/test_p116_benchmark_as_printed.py`, new, synthetic data only (10 tests): the ARMA conditional mean against
  its definition and with no look-ahead; the mean-square term against closed forms; the full forecast (variance
  part plus mean part) against a Monte Carlo of the model; P109's variants and fit keys unchanged; the 2 x 2 x 2
  factorial of specs; the driver's `mcs_ours` and `mcs_cpp` (and `mcs_cpp_varonly`) equal direct fits of
  `McsGarch` bit for bit; the Shapley split adds up and equals the one-switch effects when they are additive; Delta
  and its interval; the decision rule; and an end-to-end dry run of all three stages.

## 12. Checks run before this file was committed (3 November 2026)

* **The committed benchmark reproduces exactly through this study's driver** (REAL DATA, forecasts and targets
  only, no loss computed). The driver was run with the committed benchmark only, into a scratch folder:

  ```
  PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py run --index <INDEX> \
      --variants mcsgarch --workers 2 --work <scratch folder>
  PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p116_benchmark_as_printed.py check \
      --indexes NIFTY BANKNIFTY FINNIFTY --work <scratch folder>
  ```

  Every forecast and every target is identical to the last bit (largest relative difference 0): Nifty 19,800 of
  19,800 at 15 minutes, 19,008 of 19,008 at 30 minutes and 792 of 792 in the last hour; Bank Nifty 16,500, 15,840
  and 660; Fin Nifty 15,100, 14,496 and 604 (`y` and `mcsgarch`). Each index took about 1 minute on 2 cores after
  about 1 minute of loading the market data; every setting and the number and range of days matched the committed
  `run_info.json`.
* **P109 still reproduces with the new code path** (REAL DATA; these losses and parameters are already public in
  `results/p109_mcsgarch_extensions/`). P109's own driver, as changed here, rebuilt all 12 P109 Nifty variants
  (`p109_mcsgarch_extensions.py run --index NIFTY --workers 2`, 7.8 minutes, into a scratch folder), and P109's own
  `analyze` was run on them into a scratch folder (100 bootstrap draws, so only the files that do not depend on
  the draws were compared). Its `daily_losses_nifty_next15.csv`, `daily_losses_nifty_next30.csv` and
  `daily_losses_nifty_last_hour.csv` are byte for byte identical to the committed files (`cmp`), and all 1,140
  Nifty rows of its `params_by_refit.csv` equal the committed ones in every numeric column.
* **Dry run** (SYNTHETIC: 200 simulated days, 7 refits, two targets): all three stages ran; the rebuilt `mcs_ours`
  equalled the pipeline's walk-forward on every row (1,260 and 1,120 forecasts); stage 1 took 0.3 minutes on 2
  workers. A fit of the paper's model (ARMA mean) on 900 simulated days of 375 minutes took 3.2 s, against 1.8 s for
  `mcs_ours` (SYNTHETIC timing). With eight fits per refit we expect stage 1 to take about 10 to 20 minutes for
  Nifty and 5 to 10 minutes for each other index on 2 cores.
* **Tests.** `PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_p116_benchmark_as_printed.py`: 10
  passed. `tests/test_p109_mcsgarch_extensions.py` and `tests/test_volatility.py`: all passed. The full suite is run
  before the phase-2 commit.
