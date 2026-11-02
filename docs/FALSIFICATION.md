# Falsification tests for F1 and F2 (REAL DATA)

Everything in this document is **REAL DATA**: NSE Nifty 50 (and, in test T5, Bank Nifty) daily option
chains (bhavcopy files), 1-minute index bars, India VIX and the NSE participant-wise open interest file,
2 Jan 2023 to 2 Nov 2026. Nothing here is simulated. Every number below is copied (rounded) from a CSV
file that `experiments/falsification.py` writes; the file is named under each table.

The aim was to **try hard to break** two in-sample findings of this prototype and to add one quasi-causal
test. A null result is reported as a result, not hidden.

    OMP_NUM_THREADS=1 python experiments/falsification.py        # 303 seconds on 4 cores for the run reported here
    OMP_NUM_THREADS=1 python -m pytest -q tests/test_falsification.py

* Script: `experiments/falsification.py`. Unit tests (synthetic data only): `tests/test_falsification.py`.
* Outputs: `results/falsification/*.csv`, `results/falsification/run_info.json`, and a readable log
  `results/falsification_stdout.txt`.
* Seeds: master seed 20261102. Permutation draws use `numpy.random.default_rng([20261102, task_index])`;
  the seed of every row is in the `seed` column of `t1_permutation_summary.csv`. The sign-rule cloud uses
  `default_rng([20261102, 3])`. All of this is recorded in `run_info.json`, together with the first 16 hex
  digits of the SHA-256 of the two input files (`daily_market.csv` 959c5c633f855569,
  `daily_gex_panel.csv` 92d479816ac95eea).

## Findings

F1 is split in two parts, because they behave very differently:
**F1a** is the ordinary-day GEX coefficient (claimed negative) and **F1b** is the expiry-day interaction
(claimed positive, the "sign flip"). The main F1 measure is the scale-free **share** (net GEX / gross GEX
of the same chain); the original **ratio** (net GEX / 20-day median gross GEX) is the secondary version.
Unless stated otherwise the sign rule is **Pro** and the sample is the full sample.

| Test | F1a (ordinary days) | F1b (expiry-day flip) | F2 (reversal on negative-GEX days) |
|---|---|---|---|
| T1 permutations | **Fails** the 20-day block shuffle (p 0.246) | Survives all 5 schemes (p 0.0010 to 0.0335) | Survives all 5 schemes (p 0.0005 to 0.0100) |
| T2 timing placebos | **Fails**: GEX from the future predicts as well or better | Partly survives (wins the horse race, but leads alone are also significant) | Survives (lead t+1: p 0.31; wins the horse race) |
| T3 placebo sign rules, gross GEX, OI, calendar | **Fails** for the ratio (91.9% of random rules significant; gross GEX alone t -5.31) | Survives (no fixed rule reproduces it; rank p 0.0005), but it is carried by the daily Pro weights, not by option gamma | Survives (rank p 0.031 binary, 0.0035 with controls; gross GEX and OI do not reproduce it) |
| T4 split and out of sample | Not significant in 2023 to 2024 (p 0.13); worse forecasts | Same sign in both halves, but adding GEX makes forecasts **worse** (MSE +5.1% and +4.7%) | Survives: both halves, better forecasts 2023-24 to 2025+ (MSE -4.3%, p 0.0069), pre-registered test p 0.0059 |
| T5 Bank Nifty | Replicates (-3.07, p 0.0037) | **Does not replicate** (4.21, p 0.25) | Replicates (-0.0585, p 0.0003), but Bank Nifty is not independent evidence |
| T6 gamma roll-off (quasi-causal) | No support: under the Pro rule the roll-off has no effect (p 0.82) | (same) | Not a test of F2 |
| T7 multiple testing | Holm 0.215 among 78 original tests | Holm 0.087 among 78; 0.020 within the 18 headline tests | Holm 0.016 among 78 (binary); 0.0036 within the headline tests |

In one paragraph: **F2 under the Pro rule survived every attempt to break it**, including a test on the later
half of the sample that was specified from the earlier half. **F1's ordinary-day term did not survive**: it
behaves like a slow-moving level (gross gamma, or the trend in participant positioning), not like a daily
signal. **F1's expiry-day flip (share measure, Pro rule) survived the permutation, sign-rule and calendar
tests but has no out-of-sample forecasting value and did not replicate on Bank Nifty.** Under the US
convention neither finding exists. The quasi-causal roll-off test (T6) finds no effect that can be told apart
from placebo days. Both findings keep the same sign and similar size when the closing-auction period is
removed (sample ending 31 Jul 2026).

One finding changes the interpretation of both F1 and F2: with the Pro rule, the "GEX" regressor is driven
almost entirely by the daily Pro positioning weights from the NSE participant file. Holding those weights
fixed and letting only the option chain vary removes the F1 share result (t -0.905 and -0.504); holding the
chain fixed and letting only the weights vary reproduces it (t -2.74 and 3.26). See T3.

## Words used in this document

Each term is defined once, here.

* **Option, call, put, strike, expiry.** An option is a contract that gives the right to buy (call) or sell
  (put) the index at a fixed level (the strike) on a fixed date (the expiry). On the **expiry day** the
  contract settles and disappears.
* **Open interest (OI).** The number of option contracts outstanding at the end of a day.
* **Gamma.** How fast an option's hedge ratio changes when the index moves. Someone who is **long gamma**
  hedges by selling into rallies and buying into falls, which damps moves; someone **short gamma** does the
  opposite, which amplifies moves.
* **Dealer.** The party assumed to hedge the options it holds. Who the dealer is in India is not known; the
  sign rules below are assumptions about it.
* **GEX (gamma exposure).** For each contract: gamma x OI x spot squared x 0.01 (Black-76 gamma from the
  contract's implied volatility). **C** is the sum over all calls of the nearest expiry, **P** the sum over
  puts. **Gross GEX** = C + P (no signs). **Net GEX** = w_call x C + w_put x P, where the two weights are set by
  a sign rule.
* **Sign rule.**
  * **US convention**: w_call = +1, w_put = -1 (dealers long calls, short puts).
  * **Pro**: daily weights from the NSE participant open interest file: (Pro long minus Pro short) divided by
    the total long position, separately for index calls and index puts. These weights cover all index
    options, not only Nifty.
  * **Client mirror**: minus the same quantity computed for Clients.
* **Share measure (F1 main).** Net GEX / gross GEX of the same chain. It lies between -1 and 1 and does not
  depend on how large the chain is.
* **Ratio measure (F1 secondary, the original).** Net GEX / 20-day median of gross GEX. On expiry eves it is
  mechanically about four times larger because the front contract's gamma peaks the day before expiry.
* **Realized variance (RV).** The sum of squared 1-minute index returns over one day. F1 uses log RV.
* **India VIX.** An index of the option-implied volatility of Nifty.
* **r_rod and r_last (F2).** r_rod is the "rest of day" return (previous close to 14:59, overnight gap
  included); r_last is the return over the last 30 minutes (14:59 to 15:29). Both in basis points (bp, one
  hundredth of a percent).
* **Regime.** The expiry calendar in force: `multi_weekly`, `single_weekly_thu`, `single_weekly_tue` (as coded
  in the daily market file), plus **`closing_auction`** for days from 3 Aug 2026, when NSE sets the official
  close in an auction (the index is frozen from 15:16 to 15:28 and the 15:29 bar jumps to the auction price).
* **Samples.** **Full**: 2 Jan 2023 to 2 Nov 2026, with the closing-auction days as their own regime wherever
  regime controls or regime groups are used. **Pre-auction**: 2 Jan 2023 to 31 Jul 2026.
* **Regression, coefficient, dummy, interaction, control.** A regression fits an outcome as a weighted sum of
  explanatory variables; each weight is a coefficient. A dummy is a 0/1 variable (for example "is an expiry
  day"). An interaction is a product of two variables; its coefficient measures how the effect of one depends
  on the other. A control is a variable included so that its influence is not mistaken for the effect of
  interest.
* **F1 regression.** log RV_t on log RV_{t-1}, its 5-day mean, log VIX_{t-1}, an expiry dummy, GEX_{t-1}
  (`gex_l1`, this is F1a) and GEX_{t-1} x expiry dummy (`gex_l1_x_exp`, this is F1b), with regime x expiry
  dummies. GEX_{t-1} is from the previous evening, so it is known before day t starts.
* **F2 regression.** r_last on r_rod, a dummy for "GEX_{t-1} < 0", and r_rod x that dummy (`r_rod_x_neg`).
  A negative `r_rod_x_neg` means the last 30 minutes reverse the day's move more on negative-GEX days.
  "Binary" is this regression; "controls" adds yesterday's log VIX, r_rod x log VIX and r_rod x |r_rod|.
* **Standard error, t-statistic, p-value.** The standard error measures the noise in a coefficient; t is the
  coefficient divided by it. The p-value is the probability of a result at least this extreme if the true
  effect were zero. Small p (for example below 0.05) means the result is hard to explain as noise.
* **Newey-West (HAC) standard errors.** Standard errors that allow the errors of nearby days to be correlated
  (here up to 5 days apart). All regression p-values in this document use them.
* **Permutation test, placebo.** Replace the real GEX by a scrambled copy that cannot carry the claimed effect,
  refit, and repeat 2,000 times. The permutation p-value is (1 + number of placebo |t| at least the real |t|)
  / (1 + number of draws). A placebo is any fake version of the treatment that should show no effect.
* **Lead.** A value from the future (for example GEX published on day t+1). It cannot cause today's outcome,
  so if it "predicts" as well as the real lag, the lag's result is suspect.
* **Horse race.** A regression that includes both the real lag and the lead, to see which one carries the
  effect.
* **Out of sample.** Fit on one period, evaluate on another the model has never seen. **MSE** (mean squared
  error) measures forecast errors; lower is better. The **Diebold-Mariano (DM) test** asks whether the
  difference in MSE between two forecasts is larger than noise (one-sided p: small means "with GEX is better").
* **Pre-registered style test.** A test whose direction and specification are fixed from one period before
  looking at another period.
* **Instrumental variable (IV), two-stage least squares (2SLS).** A way to estimate a causal effect of X on Y
  when X may be confounded, using a variable Z (the instrument) that moves X for reasons unrelated to Y. The
  **first stage** regresses X on Z; its **F statistic** (here t squared with HAC errors) measures instrument
  strength (above 10 is the usual rule of thumb). The **exclusion restriction** is the assumption that Z
  affects Y only through X.
* **Multiple testing, family, Holm, BH.** When many tests are run, some will be "significant" by chance. A
  family is the set of tests adjusted together. **Holm** adjusts p-values so that the chance of even one false
  positive in the family is at most 5% (strict). **Benjamini-Hochberg (BH)** controls the expected share of
  false positives among the results declared significant (less strict).

## Replication check

Before testing anything, the script re-estimates the two original scripts' numbers from the same input files
and compares. All 48 coefficients (F1: `r3_regressions.csv`; F2: `results/momentum_gex/regressions.csv`)
match; the largest absolute difference is 2.93e-14. The script's own fast Newey-West routine is checked
against statsmodels at start-up. The option chains are also recomputed from the bhavcopy files for T3 and
T6: the front chain matches the panel within 1e-6 on 938 of 942 days, and the Pro rule rebuilt from C, P and
the participant file has a largest relative error of 5.29e-13. The 4 mismatching days are a data issue in the
panel on disk (see "Bugs and data issues").

Source: `results/falsification/replication_check.csv`, `chain_recompute_check.csv`, log line in
`results/falsification_stdout.txt`.

**Headline estimates used throughout (REAL DATA).** F1 here uses four regimes (closing auction as its own
regime), so the ratio numbers differ slightly from the original script, which has three.

| Sample | Rule | F1a share | F1b share | F1a ratio | F1b ratio | F2 binary | F2 controls |
|---|---|---|---|---|---|---|---|
| full | Pro | -3.12 (p 0.0043) | 8.18 (p 0.0013) | -2.03 (p 0.037) | 4.89 (p 0.0019) | -0.0681 (p 0.0002) | -0.0496 (p 0.0029) |
| full | US | 0.27 (p 0.37) | 0.24 (p 0.61) | -0.38 (p 0.031) | 0.39 (p 0.12) | 0.0129 (p 0.59) | -0.0008 (p 0.97) |
| full | Client mirror | -0.49 (p 0.72) | 3.10 (p 0.092) | -2.33 (p 0.014) | 3.63 (p 0.0024) | -0.0720 (p 0.00003) | 0.0026 (p 0.95) |
| pre-auction | Pro | -3.51 (p 0.0014) | 8.94 (p 0.0005) | -2.38 (p 0.013) | 5.39 (p 0.0007) | -0.0719 (p 0.00008) | -0.0526 (p 0.0013) |
| pre-auction | US | 0.37 (p 0.23) | 0.32 (p 0.52) | -0.33 (p 0.066) | 0.40 (p 0.13) | 0.0057 (p 0.81) | -0.0097 (p 0.67) |
| pre-auction | Client mirror | -0.69 (p 0.61) | 3.27 (p 0.078) | -2.67 (p 0.0057) | 3.93 (p 0.0013) | -0.0717 (p 0.00005) | 0.0042 (p 0.91) |

F1 has 937 days in the full sample and 874 before the auction; F2 has 932 and 869. (F2 here codes the 3 days with no GEX in January 2023 as non-negative, as `momentum_gex.py` did when this study ran. `momentum_gex.py` now treats them as missing, which leaves 929 days and moves the Pro binary coefficient from -0.0681 to -0.0682 with the same p-value; no conclusion here depends on it.) Source:
`results/falsification/headline_in_sample.csv`.

## T1. Permutation placebos (REAL DATA)

**What it does.** Keeps every other variable fixed and replaces GEX_{t-1} by a scrambled copy, 2,000 times per
scheme:

1. **within regime**: shuffle days inside each regime;
2. **within 20-day blocks**: shuffle days inside consecutive 20-day blocks, which keeps the slow level of GEX
   and destroys only its day-to-day variation;
3. **within regime x expiry**: shuffle inside each regime, separately for expiry and ordinary days;
4. **circular shift within regime**: rotate the GEX series by a random amount inside each regime, which keeps
   its persistence;
5. **random re-sign**: multiply each day's GEX by a random +1 or -1.

**Two-sided permutation p-values, full sample (2,000 draws each).**

| Finding | Measure | Rule | within regime | 20-day blocks | regime x expiry | circular shift | re-sign |
|---|---|---|---|---|---|---|---|
| F1a | share | Pro | 0.0065 | **0.2464** | 0.0070 | 0.0415 | 0.0055 |
| F1b | share | Pro | 0.0030 | 0.0210 | 0.0010 | 0.0335 | 0.0020 |
| F1a | share | US | 0.3933 | 0.3183 | 0.3668 | 0.3973 | 0.3918 |
| F1b | share | US | 0.6142 | 0.6292 | 0.6212 | 0.6432 | 0.6077 |
| F1a | ratio | Pro | 0.0495 | **0.6162** | 0.0455 | 0.1429 | 0.0430 |
| F1b | ratio | Pro | 0.0030 | 0.0255 | 0.0015 | 0.0475 | 0.0020 |
| F2 binary | sign | Pro | 0.0010 | 0.0050 | 0.0020 | 0.0005 | 0.0025 |
| F2 controls | sign | Pro | 0.0020 | 0.0100 | 0.0015 | 0.0015 | 0.0070 |
| F2 binary | sign | Client mirror | 0.0150 | 0.0190 | 0.0165 | 0.0115 | 0.0010 |
| F2 controls | sign | Client mirror | 0.9590 | 0.9550 | 0.9435 | 0.9545 | 0.9540 |

**Pre-auction sample, Pro rule.**

| Finding | Measure | within regime | 20-day blocks | regime x expiry | circular shift | re-sign |
|---|---|---|---|---|---|---|
| F1a | share | 0.0035 | **0.1194** | 0.0035 | 0.0245 | 0.0040 |
| F1b | share | 0.0005 | 0.0085 | 0.0015 | 0.0225 | 0.0005 |
| F1a | ratio | 0.0205 | **0.3388** | 0.0105 | 0.0785 | 0.0135 |
| F1b | ratio | 0.0015 | 0.0050 | 0.0020 | 0.0275 | 0.0015 |
| F2 binary | sign | 0.0015 | 0.0025 | 0.0015 | 0.0015 | 0.0005 |
| F2 controls | sign | 0.0020 | 0.0065 | 0.0010 | 0.0025 | 0.0015 |

**Why F1a fails the block shuffle (full sample, Pro, 20-day blocks).** The placebo keeps only the slow level of
GEX, and the placebo coefficients are almost as large as the real one:

| Term | Real coefficient | Placebo mean | Placebo 2.5% to 97.5% |
|---|---|---|---|
| F1a share | -3.124 | -2.761 | -4.289 to -1.223 |
| F1b share | 8.177 | 4.537 | 1.485 to 7.658 |
| F1a ratio | -2.031 | -2.007 | -3.324 to -0.736 |
| F1b ratio | 4.890 | 3.157 | 0.351 to 5.730 |
| F2 binary | -0.068 | -0.024 | -0.063 to 0.019 |
| F2 controls | -0.050 | -0.020 | -0.050 to 0.011 |

**Reading.** F1a is reproduced by any series with the same 20-day level, so it is a statement about slow
movements, not about yesterday's GEX. About half of F1b's coefficient is also reproduced by the slow level
(placebo mean 4.54 against 8.18), but the real value still lies outside the placebo range (p 0.021). F2 is far
from its placebo distributions under every scheme. Under circular shifts the spread of the placebo t for the
Pro F1 terms is 1.41 to 1.60 instead of 1 (column `perm_t_sd`): with such a persistent regressor the usual
Newey-West p-values are too small, which is why the permutation p-values are the ones to trust. For the
ratio F1b under 20-day blocks, the p-value computed on the coefficient instead of t is 0.102 (column
`p_two_sided_coef`); for the share F1b it is 0.012.

Source: `results/falsification/t1_permutation_summary.csv` (every draw is in `t1_permutation_draws.csv.gz`).

## T2. Timing placebos (REAL DATA)

**What it does.** Re-runs F1 and F2 with GEX published on the same evening (t, not known in time), the next
evening (t+1) or two evenings later (t+2) instead of the real t-1, and runs a horse race of t-1 against t+1.

**Full sample, Pro rule.** Coefficient (p).

| GEX published | F1a share | F1b share | F2 binary | F2 controls |
|---|---|---|---|---|
| t-1 (real) | -3.12 (0.0043) | 8.18 (0.0013) | -0.0681 (0.0002) | -0.0496 (0.0029) |
| t (same evening) | -4.51 (0.0003) | 7.20 (0.0041) | -0.0371 (0.077) | -0.0176 (0.34) |
| t+1 (lead) | -5.15 (below 0.0001) | 6.83 (0.015) | -0.0224 (0.31) | -0.0014 (0.94) |
| t+2 (lead) | -4.97 (below 0.0001) | 6.01 (0.027) | -0.0437 (0.024) | -0.0216 (0.22) |
| horse race: the t-1 term | -0.43 (0.72) | 6.42 (0.042) | -0.0812 (below 0.0001) | |
| horse race: the t+1 term | -4.87 (0.0004) | 3.21 (0.34) | 0.0262 (0.19) | |

Ratio measure (secondary), Pro: t+1 gives F1a -2.57 (p 0.0022) and F1b 3.97 (p 0.14); t+2 gives -3.17
(p 0.0002) and 3.68 (p 0.12). In the horse race the t-1 F1a term is -0.83 (p 0.44) against the lead's -2.26
(p 0.015), and the t-1 F1b term is 4.61 (p 0.015) against the lead's -0.80 (p 0.81).

**How persistent is GEX?** Autocorrelation of the Pro share: 0.800 at 1 day, 0.709 at 2 days, 0.729 at 5 days,
0.580 at 20 days. Under the Pro rule GEX is negative on 50.3% of days, and its sign is the same as the day
before on 81.8% of days. (US share: lag-1 autocorrelation 0.357, negative on 11.4% of days; Client mirror:
0.564, negative on 4.1% of days.)

**Reading.** F1a fails: GEX that is published after the outcome "predicts" volatility at least as well as the
real lag, and in the horse race the lead takes the whole effect. This is what a slow-moving level would do,
which matches T1. F1b partly survives: in the horse race the real lag keeps the expiry flip and the lead does
not, but the leads on their own also give significant flips because GEX is persistent. F2 survives: the lead
at t+1 shows nothing, and in the horse race the real lag carries the effect (-0.081) while the lead is
positive and insignificant. The one blemish for F2 is the binary t+2 lead (p 0.024; with controls p 0.22).
The same-evening GEX is not a valid predictor (it is computed after the outcome); under the US rule its F1a
coefficient is 1.01 with t 5.16, which shows how much today's move feeds into today's GEX.

Source: `results/falsification/t2_timing_placebo.csv`, `t2_gex_persistence.csv`.

## T3. Placebo sign rules, gross GEX, open interest and calendar controls (REAL DATA)

**What it does.** The option chain of every day and expiry is recomputed from the bhavcopy files
(`gex.strike_gex`), so that net GEX can be rebuilt with any weights. 2,000 random weight pairs
(w_call, w_put) are drawn uniformly from the square [-1, 1] x [-1, 1] (plus a grid of 440 pairs in steps of 0.1, without (0, 0), and 360 pairs (cos a, sin a), one per degree),
and F1 and F2 are refitted with each. If many arbitrary rules give the same result, the result is not about
who holds the options. Gross GEX and log total open interest are also tried as the regressor, and calendar
controls (dummies for 1 to 3 trading days to expiry and for weekday) are added.

**Sign-rule cloud, full sample, 2,000 uniform pairs.**

| Statistic | Share of pairs with abs(t) above 1.96 | Median abs(t) | Max abs(t) | Pro t | Pro placebo rank p | US t | Client mirror t |
|---|---|---|---|---|---|---|---|
| F1a share | 0 | 0.905 | 0.905 | -2.857 | 0.0005 | 0.905 | -0.359 |
| F1b share | 0 | 0.504 | 0.504 | 3.207 | 0.0005 | 0.504 | 1.686 |
| F1a ratio | 0.919 | 5.17 | 5.35 | -2.081 | 0.914 | -2.159 | -2.450 |
| F1b ratio | 0 | 0.281 | 1.93 | 3.108 | 0.0005 | 1.575 | 3.033 |
| F2 binary (288 usable pairs) | 0.326 | 1.59 | 4.37 | -3.702 | 0.031 | 0.544 | -4.174 |
| F2 controls (288 usable pairs) | 0 | 0.541 | 1.92 | -2.981 | 0.0035 | -0.036 | 0.067 |

"Placebo rank p" is (1 + number of pairs with a larger abs(t)) / (1 + number of usable pairs). For F2 a pair is
usable only if it has at least 20 days in each sign state; 288 of 2,000 pairs qualify (most fixed rules are
never or always negative). The grid and the angle sweep give the same picture (`t3_sign_rule_cloud_summary.csv`).

Why every fixed pair gives the same abs(t) for the share: with fixed weights, share = w_put + (w_call - w_put)
x C / (C + P), so every fixed rule is the same regressor (the call share of gamma) up to scale and sign. The
Pro and Client results can only come from the weights changing from day to day.

**Which part carries the Pro result?** Full sample, t-statistics.

| Version of the Pro rule | F1a share | F1b share | F1a ratio | F1b ratio | F2 binary | F2 controls |
|---|---|---|---|---|---|---|
| daily weights, daily chain (as published) | -2.857 | 3.207 | -2.081 | 3.108 | -3.702 | -2.981 |
| weights fixed at their mean, chain daily | -0.905 | -0.504 | -3.499 | 1.102 | only 2 negative days | |
| weights daily, chain fixed at its average shape | -2.741 | 3.262 | -2.741 | 3.262 | -3.568 | -2.821 |

**Gross GEX and open interest as the regressor (full sample).**

| Regressor | F1a | F1b | F2 (below-median dummy) | F2 with controls |
|---|---|---|---|---|
| gross GEX / 20-day median | -0.271 (t -5.31, p below 0.0001) | 0.004 (p 0.97) | -0.0124 (p 0.54) | 0.0118 (p 0.53) |
| log gross GEX | -0.204 (t -4.46, p below 0.0001) | 0.151 (p 0.077) | 0.0062 (p 0.79) | 0.0208 (p 0.30) |
| log total OI | -0.110 (p 0.053) | 0.348 (p 0.16) | -0.0127 (p 0.60) | 0.0090 (p 0.67) |

**Calendar controls (full sample).**

| Regression | Without calendar controls | With calendar controls |
|---|---|---|
| F1a share, Pro | -3.12 (p 0.0043) | -4.08 (p 0.0004) |
| F1b share, Pro | 8.18 (p 0.0013) | 8.80 (p 0.0005) |
| F1b share, Client mirror | 3.10 (p 0.092) | 4.23 (p 0.021) |
| F1a ratio, US | -0.38 (p 0.031) | -0.23 (p 0.22) |
| F1a, gross GEX / median | -0.271 (p below 0.0001) | -0.465 (p below 0.0001) |
| F2 binary, Pro | -0.0681 (p 0.0002) | -0.0604 (p 0.0004) |
| F2 controls, Pro | -0.0496 (p 0.0029) | -0.0452 (p 0.0072) |

**Reading.** The original ratio F1a is not about the sign of dealer gamma: 91.9% of random rules give a
significant coefficient, most of them larger than Pro's, and gross gamma alone gives t -5.31. F1b and the share
version of F1 cannot be produced by any fixed rule, so in that sense they survive, but the decomposition shows
that they are carried by the daily Pro weights (the participant positioning series), not by the option chain's
gamma. For F2 under Pro, 2.8% of usable random rules give a larger abs(t) (binary) and none with controls; F2 is not
reproduced by gross GEX or OI, and is also carried by the weights (t -3.57 with the chain fixed). Calendar
controls do not remove F1 (share, Pro) or F2. The pre-auction sample gives the same picture
(`t3_*.csv`, rows with `sample = pre_auction`).

Source: `results/falsification/t3_sign_rule_cloud.csv` (every pair), `t3_sign_rule_cloud_summary.csv`,
`t3_real_rules_in_cloud.csv`, `t3_gross_and_oi.csv`, `t3_calendar_controls.csv`, and the recomputed chains in
`chain_gex_by_expiry_nifty.csv.gz`.

## T4. Out-of-sample split and a pre-registered style test (REAL DATA)

**What it does.** Splits the sample at 31 Dec 2024: A = 2023 to 2024 (485 F1 days), B = 2025 onward (452 F1
days; 389 before the auction). Estimates each half separately, then fits on one half and forecasts the other,
with and without GEX. Finally a one-sided test of F2 on period B only, with the direction and specification
(Pro, binary, all days) taken from period A.

**Split estimates, Pro rule, full sample.** Coefficient (p).

| Term | 2023 to 2024 | 2025 onward | 2025 onward, pre-auction |
|---|---|---|---|
| F1a share | -2.67 (0.13) | -3.62 (0.013) | -4.36 (0.0038) |
| F1b share | 10.46 (0.0029) | 6.94 (0.081) | 8.16 (0.045) |
| F1a ratio | 0.24 (0.87) | -3.96 (0.0027) | -4.69 (0.0003) |
| F1b ratio | 4.53 (0.044) | 5.65 (0.014) | 6.50 (0.0055) |
| F2 binary | -0.0644 (0.019) | -0.0739 (0.012) | -0.0796 (0.0069) |
| F2 controls | -0.0405 (0.13) | -0.0702 (0.0003) | -0.0742 (below 0.0001) |
| Pro negative-GEX days (F2) | 365 of 480 | 98 of 452 | 90 of 389 |

**Out-of-sample forecasts, Pro rule.** Change in MSE when GEX is added (negative = better), and the one-sided
DM p-value for "GEX helps".

| Finding | Fit 2023-24, test 2025+ | Fit 2025+, test 2023-24 | Same, pre-auction (A to B / B to A) |
|---|---|---|---|
| F1 share | +5.12% (p 0.98) | +4.75% (p 0.97) | +4.15% (p 0.94) / +6.31% (p 0.98) |
| F1 ratio | +4.32% (p 0.97) | +7.17% (p 0.998) | +4.39% (p 0.96) / +9.09% (p 0.999) |
| F2 | -4.30% (p 0.0069) | -2.09% (p 0.23) | -5.77% (p 0.0006) / -1.98% (p 0.24) |

**Pre-registered style test of F2 on 2025 onward.** Hypothesis from 2023 to 2024: `r_rod_x_neg` is negative
(estimate there -0.0644).

| Sample | Rule | Coefficient 2025 onward | t | One-sided p | Days | Negative-GEX days | Passes at 5% |
|---|---|---|---|---|---|---|---|
| full | **Pro (primary)** | -0.0739 | -2.52 | 0.0059 | 452 | 98 | yes |
| pre-auction | **Pro (primary)** | -0.0796 | -2.70 | 0.0035 | 389 | 90 | yes |
| full | US | 0.0145 | 0.56 | 0.71 | 452 | 71 | no |
| full | Client mirror | -0.1042 | -2.45 | 0.0072 | 452 | 6 | yes (only 6 negative days) |

**Reading.** F1's expiry flip has the same sign in both halves and is significant in at least one version in
each, but adding GEX makes F1's forecasts worse in both directions and in both samples: F1 has no
out-of-sample value. F1a is not even significant in 2023 to 2024. F2 holds in both halves, improves the
forecast of 2025 onward from a model fitted on 2023 to 2024, and passes the pre-registered style test. The
reverse direction does not reach significance (p 0.23). Note that the Pro negative-GEX state is concentrated
in 2023 to 2024 (365 of 480 days) and much rarer later (98 of 452), so "negative GEX" is partly a period
marker; the result inside each period is what makes F2 credible. The split date was chosen by me after the
full-sample result was known, so this is "pre-registered style", not a true pre-registration.

Source: `results/falsification/t4_split_estimates.csv`, `t4_oos_forecasts.csv`, `t4_preregistered_f2.csv`.

## T5. Bank Nifty with the same machinery (REAL DATA)

**What it does.** Builds the Bank Nifty market and GEX panel with the same functions
(`build_market_from_nse`, `build_gex_panel`) and runs F1 and F2 unchanged. Bank Nifty data run from
2 Jan 2023 to 22 Apr 2026: 810 days, 115 expiry days (98 multi_weekly, 10 single_weekly_thu,
7 single_weekly_tue). There are no closing-auction days in this sample.

| Finding | Measure | Rule | Coefficient | p | Days |
|---|---|---|---|---|---|
| F1a | share | Pro | -3.07 | 0.0037 | 805 |
| F1b | share | Pro | 4.21 | 0.25 | 805 |
| F1a | ratio | Pro | -1.76 | 0.025 | 805 |
| F1b | ratio | Pro | 3.84 | 0.038 | 805 |
| F1a / F1b | share | US | 0.38 / -0.28 | 0.27 / 0.70 | 805 |
| F2 binary | sign | Pro | -0.0585 | 0.0003 | 809 (462 negative) |
| F2 controls | sign | Pro | -0.0533 | 0.0020 | 808 |
| F2 binary | sign | US | 0.0086 | 0.73 | 809 |
| F2 binary | sign | Client mirror | -0.0478 | 0.017 | 809 (37 negative) |
| plain momentum (r_rod only) | | none | -0.0255 | 0.0058 | 809 |

**How independent is this second market?** Not very. On the 800 days the two F2 samples share, the
correlation of r_last between the indices is 0.833 and of r_rod 0.877. The sign of GEX is the same for both
indices on 98.2% of days under the Pro rule (99.9% Client mirror, 84.1% US), because the participant weights
are the same series for both (they cover all index options).

**Reading.** F2 replicates on Bank Nifty, but this is close to re-using the Nifty test: same weights, highly
correlated returns. F1's expiry flip does not replicate with the share measure (p 0.25); the ratio version
does (p 0.038). F1a replicates, which fits the T1 and T2 picture of a slow-moving level shared by both indices.

Source: `results/falsification/t5_banknifty.csv`, `banknifty_gex_panel.csv`, `banknifty_daily_market.csv`,
`run_info.json` (key `banknifty`).

## T6. Quasi-causal test: gamma roll-off on expiry days (REAL DATA)

**Idea.** On an expiry day the front contract's gamma leaves the market at the close for a calendar reason,
whatever happened that day. If dealer gamma damps volatility, losing a large slice of positive net gamma should
be followed by higher volatility the next day. Days on which nothing expires give placebos.

**Design.** For each event day p, with q the trading day before:

* `shock` = net GEX of the contracts that expire on p, measured at the close of q;
* `rest` = net GEX of all later expiries at the close of q;
* both divided by the 20-day median of all-expiry gross GEX at q, with the sign weights of q.

Main regression: log RV on the day after p, on `shock`, `rest`, log RV_q, its 5-day mean, log VIX_q and regime
dummies. The event day itself is skipped as an outcome because its own volatility is affected by the
settlement. **Prediction under damping: coefficient on `shock` above 0.** IV version: the change in net GEX
across p (after minus before) instrumented by `shock`, controlling for total GEX before; **prediction: IV
coefficient below 0.**

Placebo event sets: 2 trading days before and after each expiry (clean), 1 day before and after
(contaminated, because the outcome day or the pre-event day is the real expiry day; reported but not used for
conclusions), and the other regime's expiry weekday. Nifty has 200 real events (199 used in the regression),
197 at -2, 197 at +2 and 172 weekday placebos; Bank Nifty has 115 real events (114 used).

**Nifty, full sample, main specification.** Coefficient on `shock` per one standard deviation of `shock`
(p); IV estimate (p) and first-stage F.

| Rule | Real expiry days | Placebo -2 | Placebo +2 | Placebo weekday | IV (dGEX) | First-stage F |
|---|---|---|---|---|---|---|
| US | +0.083 (0.029) | +0.002 (0.95) | +0.013 (0.71) | +0.007 (0.84) | -5.83 (0.020) | 48.8 |
| Pro | -0.046 (0.82) | -0.017 (0.90) | +0.031 (0.78) | +0.034 (0.75) | 1.03 (0.97) | 382 |
| Client mirror | -0.204 (0.012) | -0.181 (0.0034) | -0.086 (0.24) | -0.101 (0.11) | 20.46 (0.042) | 668 |
| gross | -0.031 (0.66) | -0.117 (0.0044) | -0.136 (0.024) | -0.060 (0.21) | 0.38 (0.56) | 659 |

**Real minus placebo, Nifty full sample** (z-test on the per-SD coefficients; p in brackets).

| Rule | vs -2 | vs +2 | vs weekday |
|---|---|---|---|
| US | 1.67 (0.094) | 1.33 (0.18) | 1.49 (0.135) |
| Pro | -0.12 (0.90) | -0.33 (0.74) | -0.35 (0.73) |
| Client mirror | -0.22 (0.83) | -1.08 (0.28) | -1.01 (0.31) |
| gross | 1.06 (0.29) | 1.14 (0.26) | 0.34 (0.73) |

Other checks: US real coefficient with the expiry-day RV added as a control 0.726 (p 0.032), with a
monthly-expiry dummy 0.753 (p 0.023). Pre-auction, US real +0.086 per SD (p 0.039), IV -6.33 (p 0.020), real
minus placebo z 1.54, 1.30 and 1.39 (p 0.12 to 0.19). The contaminated day-before placebo is large under Pro
(+0.353 per SD, p 0.0007), which is why it is not used. Bank Nifty: US real +0.018 per SD (p 0.78); Pro
-0.327 per SD (p 0.054), IV +40.4 (p 0.097), the opposite of the damping prediction.

**Reading.** Under the US rule the roll-off moves volatility in the direction the damping story predicts, and
the IV estimate is significant at 5%. But the real-day effect cannot be distinguished from the clean placebo
days (p 0.094 to 0.18), it does not survive the multiple-testing adjustment (BH 0.094 among the 742 tests of
this study, Holm 1.0), and it is absent on Bank Nifty. Under the Pro rule, the one that F1 and F2 rest on, the
roll-off has no effect at all. Under the Client mirror the coefficient is negative, but the -2 placebo is just
as negative, so it reflects something other than the roll-off. **Verdict: no robust evidence that removing
gamma mechanically changes next-day volatility; the test gives no support to the Pro-rule mechanism behind
F1.**

**What T6 identifies, and what it does not.**

* It identifies the effect on next-day log RV of the calendar-timed disappearance of the expiring contracts'
  net gamma, holding fixed the gamma that remains, recent volatility, VIX and the regime, under the chosen
  sign rule. The timing of the loss is set by the exchange calendar, not by traders.
* The size of the expiring gamma is not random. It is predetermined at q, but traders chose it, and weekly
  open interest builds up during the week; weeks with more positioning may be weeks with different expected
  volatility.
* Gamma is inversely related to implied volatility, so a large shock can simply mean a calm market at q. This
  is handled only through controls (VIX_q and RV_q) and the placebo days.
* The exclusion restriction of the IV can fail: the expiry day has its own volatility pattern, which can carry
  into the next day through persistence. Adding expiry-day RV as a control (a post-treatment variable, so
  itself possibly biased) changes little.
* A null under a wrong sign rule is not evidence against gamma effects in general.
* The real-minus-placebo z-test treats the real and placebo estimates as independent. They share weeks, so
  their covariance is most likely positive, the true standard error of the difference is smaller, and the
  z-test is conservative.

Source: `results/falsification/t6_rolloff_regressions.csv`, `t6_rolloff_iv.csv`, `t6_real_minus_placebo.csv`,
`t6_event_table_nifty.csv` (one row per event and rule), and `chain_gex_by_expiry_banknifty.csv.gz`.

## T7. Multiple testing (REAL DATA)

**What it does.** Counts the tests and adjusts the p-values in three families.

* **Family A**: every GEX-term test the original scripts report (R3 regressions, the R3 robustness table and R5),
  78 tests. Numbers in this family come from the original scripts as they are on disk (three regimes), so the
  ratio coefficients differ slightly from the four-regime numbers elsewhere in this document.
* **Headline set**: the 12 original headline claims (F1 ratio and F2, three rules) plus the 6 share rows,
  18 tests, also adjusted on their own.
* **Family B**: every test this script reports, 742 tests.

**Headline tests, Pro rule (full sample).**

| Test | Coefficient | Raw p | Holm (78) | BH (78) | Holm (18) | BH (18) |
|---|---|---|---|---|---|---|
| F1a share | -3.158 | 0.0037 | 0.215 | 0.014 | 0.044 | 0.0094 |
| F1b share | 8.209 | 0.0013 | 0.087 | 0.0085 | 0.020 | 0.0074 |
| F1a ratio | -2.068 | 0.033 | 1.0 | 0.078 | 0.30 | 0.059 |
| F1b ratio | 4.929 | 0.0017 | 0.107 | 0.0086 | 0.025 | 0.0074 |
| F2 binary | -0.0681 | 0.0002 | 0.016 | 0.0056 | 0.0036 | 0.0019 |
| F2 controls | -0.0496 | 0.0029 | 0.172 | 0.012 | 0.037 | 0.0086 |
| F2 binary, Client mirror | -0.0720 | 0.00003 | 0.0023 | 0.0012 | 0.0005 | 0.0005 |
| F1b share, US | 0.234 | 0.62 | 1.0 | 0.78 | 1.0 | 0.75 |

Before the auction, F1b share (Pro) has Holm 0.039 within family A (raw p 0.0005).

**Headline set with permutation p-values** (within-regime shuffle, 12 tests, Holm and BH).

| Test | Permutation p | Holm | BH |
|---|---|---|---|
| F1a share, Pro | 0.0065 | 0.0585 | 0.0195 |
| F1b share, Pro | 0.0030 | 0.030 | 0.012 |
| F2 binary, Pro | 0.0010 | 0.012 | 0.012 |
| F2 controls, Pro | 0.0020 | 0.022 | 0.012 |
| F2 binary, Client mirror | 0.0150 | 0.120 | 0.036 |
| F1b share, Client mirror | 0.0960 | 0.672 | 0.192 |

**Family B (742 tests).** 279 have raw p below 0.05, 146 have BH below 0.05 and 16 have Holm below 0.05. Of
those 16, 12 are placebo or challenger regressions, mostly bearing on F1a (Pro GEX leads at t+1 and t+2,
same-evening US GEX, gross GEX with and without calendar controls); 4 support a finding (the F2 Pro horse race in both samples, F2 Pro with
controls in 2025 onward before the auction, and F1b ratio Pro with calendar controls before the auction). The
pre-registered F2 test has BH 0.035 (pre-auction 0.025) in this family.

**Reading.** Among the Pro-rule headline tests, only F2 (binary) survives Holm in the full family of the original tests. The Client-mirror version of F2 also survives (Holm 0.0023), but it rests on 38 negative-GEX days and disappears once VIX and the size of the day's move are controlled (R5), so it is not counted as support for a dealer effect. F1b survives Holm
only within the narrow headline set. With 742 tests in this study alone, isolated p-values near 0.02 should be
read as weak evidence.

Source: `results/falsification/t7_family_A_original_tests.csv`, `t7_headline_adjusted.csv`,
`t7_headline_permutation_adjusted.csv`, `t7_family_B_all_tests_this_study.csv`.

## Limitations

* **One market, short sample.** About 940 days and 199 expiry days of Nifty. Bank Nifty is not an independent
  replication (same participant weights, returns correlated 0.83 to 0.88).
* **The sign rules are assumptions.** "Pro" in the NSE file means proprietary traders, not necessarily the
  dealers who hedge, and the weights cover all index options and all expiries, not the Nifty front chain.
  T3 shows that the Pro results come from this positioning series rather than from option gamma, so a
  reading of F1 or F2 as "dealer gamma" is not supported by these tests; a reading as "Pro positioning" is.
* **Persistence.** GEX under the Pro rule is very persistent (lag-1 autocorrelation 0.80), so every
  regression p-value that assumes a well-behaved regressor is too optimistic (T1 circular shifts). The
  permutation p-values are the better guide.
* **Period effects.** Negative Pro GEX is common in 2023 to 2024 and rare later; F2 holds inside each period,
  but a structural change around the same time could still matter.
* **The split and the "pre-registered" test** were designed after the full-sample results were known.
* **Closing auction.** Only 63 days (3 Aug to 2 Nov 2026). Treating them as a separate regime and dropping them
  both leave the results unchanged in sign and similar in size, but the period is too short to study on its
  own.
* **T6** cannot rule out that the size of the expiring gamma reflects trader choices correlated with future
  volatility, and its exclusion restriction can fail (see the list in T6).
* **F2 coding inherited from the original script.** Days with unknown GEX are coded as non-negative (3 days in
  Jan 2023) and the VIX control is lagged after dropping rows; this is kept for comparability (see below).
  T2's lead placebos do not use the first coding.

## Bugs and data issues found (not fixed here)

1. **Stale front expiry on 4 days in the GEX panel.** `results/pipeline_nse/daily_gex_panel.csv` (written
   20:33 on 2 Nov 2026) still uses the listed expiry 2023-06-29 on 2023-06-22, 06-23, 06-26 and 06-27, but the
   contracts expired on 2023-06-28 (holiday move). `src/niftygex/data.py` now has `holiday_adjusted_expiries`,
   which the panel predates. The effect is small (relative differences in C and P of 0.00009 to 0.00057).
   Reproduce: rows with `remapped = True` and non-zero `rel_diff_C` in
   `results/falsification/chain_recompute_check.csv`. Fix: rebuild the panel with `run_pipeline`.
2. **`experiments/momentum_gex.py` line 102** (`neg = (g < 0).astype(float)`, also line 135): a missing GEX
   compares as False, so 3 days without a GEX value (2023-01-03, 01-04, 01-05) are kept and coded as
   non-negative instead of being dropped.
3. **`experiments/momentum_gex.py` line 89**: `lvix_prev` is computed with `.shift(1)` after the
   `dropna` on line 86, so on the 4 days after a dropped day (2023-01-10, 2023-02-06, 2023-04-13,
   2023-04-28) the "previous" VIX is from two or more trading days earlier.

None of the three changes any conclusion; this script reproduces the original numbers exactly (see Replication check) and keeps
items 2 and 3 for comparability.

## Files

* `experiments/falsification.py`: the script (all seven tests).
* `tests/test_falsification.py`: 10 unit tests on synthetic data (Newey-West against statsmodels, collinear
  dummies, IV, permutation helpers, the expiring-gamma split, placebo dates).
* `results/falsification/`: replication_check, headline_in_sample, t1 to t7 tables, recomputed chains, Bank
  Nifty panel and market files, run_info.json.
* `results/falsification_stdout.txt`: the full printed log of the run.
