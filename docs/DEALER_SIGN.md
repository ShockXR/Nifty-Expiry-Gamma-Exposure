# Estimating the dealer sign from data (P82)

**REAL DATA** (NSE option chains, NSE participant-wise open interest, validated 1-minute Nifty bars,
6 Feb 2023 to 2 Nov 2026), except the section marked **SYNTHETIC**, which uses simulated outcomes on the
real regressors. Every number below is read from a file written by `experiments/dealer_sign.py`; the
file is named under each table. How to rerun it is at the end.

---

## Findings

This is a clear negative result. In plain terms: the data do not reveal one dealer sign that fits both
predictions of hedging theory, and a sign estimated from the data does not forecast better than the three
fixed guesses.

1. **The two outcomes point in nearly opposite directions.** When dealers are described as a mix of the
   Pro rule and the Client-mirror rule (angle 0 degrees = Pro, 90 = Client mirror, 180 = reverse Pro,
   270 = Client), the variance outcome (a) points to **21 degrees** (mostly Pro), and the end-of-day
   momentum outcome (b) points to **190 degrees** (reverse Pro). The two are 169 degrees apart. Outcome (b)
   is strong (Wald p = 4.3e-6; 98.5% of bootstrap draws within 90 degrees of 190; shuffled-outcome placebo
   p = 0.005). Outcome (a) is weak (Wald p = 0.18; placebo p = 0.144). The joint estimate, 190 degrees
   (w = -0.985 Pro - 0.174 Client mirror), is therefore driven by (b) alone. This repeats the earlier
   tension in the project (R3 fits the Pro sign, R5 fits the reversed Pro sign), now with a scale-free GEX
   measure and an explicit estimator.
2. **The call/put split reveals only one bit, and that bit is not reliable.** With GEX divided by gross
   gamma of the same chain, only the sign of w_call - w_put is identified. Outcome (b) prefers the US side
   (w_call > w_put; full-sample p = 0.036), outcome (a) is silent (p = 0.45), the joint estimate is the US
   convention, but only 63.5% of bootstrap draws land on that side and the placebo p is 0.139. In rolling
   250-day windows the side flips to reverse US in 2026 (in the last five windows, ending June to October
   2026, only 1.5% to 6% of bootstrap draws are on the US side).
3. **Out of sample, no estimated rule significantly beats any fixed rule.** In an expanding window refitted every 21
   trading days over 452 test days (2025-01-01 to 2026-11-02), all 10 primary Diebold-Mariano tests have
   Holm-adjusted p = 1.0. The best case is the participant estimate in outcome (b): 2.3% lower squared error
   than the no-dealer baseline (DM p = 0.25), 3.3% lower than the US rule (p = 0.137), 0.8% lower than Pro
   (p = 0.41). That 2.3% gain is larger than in 97% of shuffled-outcome placebos (placebo p = 0.030), so it
   is probably not pure chance, but it is too small to separate from the fixed rules. In outcome (a) the
   participant estimate is worse than the baseline by 1.0%.
4. **The one significant out-of-sample gain uses the Pro rule with the anti-theory sign.** Letting the Pro
   GEX coefficient take any sign in outcome (b) lowers squared error by 5.2% versus the baseline (DM
   p = 0.0027). Its fitted sign is the opposite of what hedging theory says for the Pro rule at all 22
   refits. Under the maintained hypothesis this is the same evidence as finding 1: in (b) dealers look like
   reverse Pro.
5. **The estimated sign agrees little with the fixed rules.** On test days the estimated participant GEX is
   negative (dealers short gamma) on 93.8% of days and has the same sign as Pro GEX on 15.9%, as
   Client-mirror GEX on 5.3% and as US GEX on 19.7%.
6. **The weights are not stable.** The participant direction moved from 272 degrees (training, close to
   Client) to 207 degrees at the May 2025 refit and stayed between 189 and 201 degrees afterwards. Its 90%
   bootstrap bands stay wide (61 to 314 degrees in the full sample; 169 to 339 degrees wide in the rolling
   windows ending from May 2025 on). The 2023 to 2024 LR set (194 to 328) and the full-sample LR set
   (178 to 217) barely overlap.
7. **The method works when the signal is strong enough (SYNTHETIC).** On the real regressors with known
   weights, the 2-D participant direction is found with a median error of 9.0 degrees and detected 95% of
   the time at signal-to-noise 0.10; at 0.05 the error is 22.2 degrees and detection 35.5%. The fixed rules'
   in-sample signal-to-noise on the real data (a roughly comparable measure) is 0.035 to 0.069 in (a) and
   0.14 to 0.16 in (b), so (a) sits in the low-power range and (b) where the method should work.
8. **The old R3 scaling creates a false direction.** With net GEX divided by the 20-day median of gross
   (gross GEX at the close before an expiry day is 1.71 times its 20-day median, against 0.52 four trading
   days before expiry), the call/put shares appear to reveal a 2-D direction in (a) (52 degrees, Wald
   p = 0.013). With the scale-free measure that
   direction is not identified at all. The scale-free measure is used throughout.

---

## Words used here

- **GEX (gamma exposure)**: how much dealers must buy or sell the index for a 1% move to stay hedged.
  Positive GEX (dealers long gamma) means they sell rallies and buy dips, which should calm the market.
  Negative GEX means they chase moves.
- **State-space model**: a regression whose coefficients are allowed to change slowly over time. The
  coefficients are hidden "states"; each day they move by a small random step, and the data are noisy
  measurements of them.
- **Kalman filter**: the standard recipe for a state-space model. Each day it updates its best guess of
  the hidden coefficients from yesterday's guess and today's data, and tracks how uncertain that guess is.
  Here the size of the daily step (q) is estimated by maximum likelihood on 2023 to 2024.
- **Identification**: whether the data can, in principle, pin down a quantity. Open interest alone cannot
  say who hedges; the dealer weights are identified only through an assumption about how hedging moves
  prices (below).
- **Out of sample**: a forecast scored on days that were not used to fit it. Here every forecast for a day
  uses only data from before the day's 21-day block.
- **Diebold-Mariano (DM) test**: a test of whether two forecasts have the same average loss, using the
  day-by-day loss differences and allowing them to be correlated over time (here with the
  Harvey-Leybourne-Newbold small-sample correction from `src/niftygex/evaluation.py`).
- **QLIKE**: a loss for variance forecasts that is robust to noise in realized variance. Lower is better.
- **LR set**: all angles whose likelihood-ratio gain is within 2.706 (the 90% chi-square value with one
  degree of freedom) of the best angle.
- **Block bootstrap**: resampling blocks of 10 consecutive days with replacement and re-estimating, to see
  how much the estimate moves.

---

## Method

### Data

913 of 942 days have every input (2023-02-06 to 2026-11-02). 461 training days (before 2025-01-01), 452
test days. 194 are expiry days. From 3 Aug 2026 NSE runs a closing auction that freezes the minute index
from 15:16 to 15:28 and jumps at 15:29, so the 63 auction days are left out of the end-of-day outcome (b)
in the main run (389 test days for (b)). Source: `results/dealer_sign_stdout.txt`.

### The GEX measure is linear in the weights and scale-free

Write G_call and G_put for the summed gamma of all calls and all puts in the day's chain. Any dealer rule
gives net GEX = w_call x G_call + w_put x G_put, so GEX is linear in the weights (checked to 2.1e-17 against
the project's own Pro GEX). Every candidate is divided by gross GEX of the same chain, which gives a number in
[-1, 1]; log gross GEX relative to its 20-day median is a control in every model. GEX inputs are from the
close of the day before the outcome.

Two families of weights:

| Family | GEX / gross | Fixed rules on the circle | What can be identified |
|---|---|---|---|
| shares | w_call x s_call + w_put x s_put, with s_call + s_put = 1 | US convention = 315 degrees (1, -1); reverse US = 135 | only the sign of w_call - w_put |
| participants | a x Pro + b x ClientMirror | Pro = 0, Client mirror = 90, FII+DII = 135, reverse Pro = 180, Client = 270 | a 2-D direction |

Because s_call + s_put = 1, the shares family collapses to one series (US GEX / gross), so a 2-D
direction there is not identified; the estimator reports only the side. The call share of gross gamma has
mean 0.546 and standard deviation 0.039.

### The identifying assumption

**Maintained hypothesis: hedging theory is right. (a) On days that are not expiry days, next-day log
realized variance, beyond a HAR baseline (yesterday, last 5 days, last 22 days, India VIX, log gross GEX),
falls when dealer gamma is higher. (b) The end-of-day momentum slope (last-30-minute return regressed on the
rest-of-day return) rises when dealers are short gamma, that is, the coefficient on rest-of-day return x GEX
is negative.**

The weights are chosen to make these predictions fit best. The result is a **revealed sign under a
maintained hypothesis**: it says which dealer rule would make hedging theory true, not who the dealers
actually are. If the theory is wrong, or another force drives the same patterns, the revealed sign is
wrong too.

### Estimator

Only the direction of the weights is identified (a longer weight vector just gives a smaller regression
coefficient), so w = (cos phi, sin phi) and phi is scanned on a 1-degree grid. For each phi the script adds
GEX(w) x multiplier (multiplier = 1 - expiry dummy in (a), rest-of-day return in (b)) to the baseline, with
the coefficient forced to the theory sign (negative); a wrong-sign fit counts as no gain. The family's GEX
series also enter freely (times the expiry dummy in (a), as levels in (b)), so the direction comes only from
the theory-signed term. The fit gain is a Gaussian likelihood ratio (LR); the joint objective adds the LR of
(a) and (b). Uncertainty: the 90% LR set, a 200-draw block bootstrap (blocks of 10 days) and a HAC Wald test
of "no GEX effect" on the unrestricted loadings.

The LR set is not trustworthy on its own: searching over angles inflates the LR even with no effect
(placebo median 5.0 for the participant joint LR, far above 2.706), and the null synthetic coverage is 0.67.
The bootstrap share of draws within 90 degrees of the estimate, and the placebo, are the main uncertainty
measures.

### Weight stability

A 250-day rolling window (32 windows, each with bootstrap bands), the expanding out-of-sample path, and a
time-varying-parameter state-space model: the two GEX loadings follow random walks, the controls are fixed,
and q (the daily step variance) is estimated by maximum likelihood on 2023 to 2024 with an LR test of
q = 0. Figures: `results/dealer_sign/figures/dealer_weights_over_time.png` (the fixed rules are drawn as
dashed and dotted lines), `dealer_weights_kalman.png`, `lr_profiles.png`.

### Out-of-sample contest

Test period 2025-01-01 to 2026-11-02, expanding window, refit every 21 trading days (22 refits). At each
refit the direction is re-estimated on all earlier days, then each candidate's forecasting regression is fit
and used for the next block:

- baseline (no dealer sign; it already contains log gross GEX);
- the three fixed rules (US, Pro, Client mirror);
- est_shares and est_participants (direction re-estimated at each refit), plus "frozen" versions that keep
  the first (2023 to 2024) direction;
- Kalman versions (free template only).

Each GEX enters twice: (a) GEX x (1 - expiry) [theory-signed] and GEX x expiry [free]; (b) GEX level [free]
and rest-of-day return x GEX [theory-signed]. In the **signed** template (primary) a theory-signed
coefficient with the wrong sign is set to 0. In the **free** template it can take any sign, so a rule and
its reverse forecast identically. Losses: QLIKE on realized variance (with a smearing correction) for (a);
squared error of the last-30-minute return in bp^2 for (b). The primary tests are the 10 DM tests of
estimated sign against each fixed rule in the signed template, Holm-adjusted.

### Placebo and synthetic check

Placebo: 200 shuffles of whole outcome rows (outcome, its HAR lags, VIX, expiry and auction flags) against
the GEX series in calendar order, then the whole estimation and the out-of-sample contest rerun (20
shuffles for the Kalman model). Synthetic: outcomes simulated on the real regressors with known weights at
several signal-to-noise ratios (signal standard deviation / noise standard deviation), 200 draws each.

---

## Results

### 1. Revealed directions (REAL DATA)

Source: `results/dealer_sign/direction_estimates.csv`. Angle in degrees. Boot band = 90% block-bootstrap
band (wraps through 0 when the first number is larger). Same side = share of bootstrap draws within 90
degrees of the estimate. Wald p = HAC test that both GEX loadings are 0.

| Sample | Family | Objective | Angle | LR | 90% LR set | Boot band | Same side | Wald p |
|---|---|---|---|---|---|---|---|---|
| 2023 to 2024 | shares | joint | 315 (US) | 11.40 | 226 to 44 | 135 to 315 | 0.705 | |
| 2023 to 2024 | shares | (a) | 135 | 1.81 | all angles | | 0.800 | 0.36 |
| 2023 to 2024 | shares | (b) | 315 | 11.40 | 226 to 44 | | 0.850 | 0.0019 |
| 2023 to 2024 | participants | joint | 272 | 11.02 | 194 to 328 | 186 to 69 | 0.515 | |
| 2023 to 2024 | participants | (a) | 29 | 8.30 | 359 to 90 | 338 to 90 | 0.990 | 0.058 |
| 2023 to 2024 | participants | (b) | 272 | 11.02 | 194 to 328 | 158 to 19 | 0.810 | 0.0029 |
| full sample | shares | joint | 315 (US) | 7.98 | 226 to 44 | 135 to 315 | 0.635 | |
| full sample | shares | (a) | 135 | 0.82 | all angles | | 0.815 | 0.45 |
| full sample | shares | (b) | 315 | 7.98 | 226 to 44 | | 0.770 | 0.036 |
| full sample | participants | joint | 190 | 21.28 | 178 to 217 | 61 to 314 | 0.850 | |
| full sample | participants | (a) | 21 | 3.75 | 331 to 113 | 318 to 116 | 0.925 | 0.18 |
| full sample | participants | (b) | 190 | 21.28 | 178 to 217 | 158 to 241 | 0.985 | 4.3e-6 |

Reading the table: in the full sample, the (a) LR set contains Pro (0) and Client mirror (90) but not reverse
Pro; the (b) and joint sets contain reverse Pro (180) and none of the fixed rules
(`in_set:` columns of the same file). Unrestricted loadings in the full sample: (a) theta = (-1.29, -0.51),
standard errors (1.32, 1.71); (b) theta = (1.34, 0.24), standard errors (0.62, 0.26). The revealed direction
is minus theta, so (a) says "Pro-like" and (b) says "reverse Pro".

### 2. The fixed rules inside the same regressions (REAL DATA, in sample)

Source: `results/dealer_sign/fixed_rule_fits.csv`. Coefficient on the theory-signed term (theory says
negative), HAC standard errors, two-sided p.

| Sample | Outcome | US convention | Pro | Client mirror |
|---|---|---|---|---|
| 2023 to 2024 | (a) | +0.52 (p = 0.36) | -4.47 (p = 0.020) | -2.55 (p = 0.15) |
| 2023 to 2024 | (b) | -0.33 (p = 0.0019) | +1.45 (p = 0.12) | +0.69 (p = 0.0006) |
| full sample | (a) | +0.22 (p = 0.45) | -1.64 (p = 0.066) | -1.16 (p = 0.35) |
| full sample | (b) | -0.20 (p = 0.036) | +1.68 (p = 0.00023) | +0.56 (p = 0.0021) |
| full, regime controls | (a) | +0.18 (p = 0.52) | -2.71 (p = 0.009) | -1.41 (p = 0.27) |
| full, regime controls | (b) | -0.21 (p = 0.034) | +2.10 (p = 0.00014) | +0.68 (p = 0.00001) |

No fixed rule has the theory sign in both outcomes. Pro and Client mirror have it in (a) only; the US
convention in (b) only.

### 3. Out of sample against the no-dealer baseline (REAL DATA)

Source: `results/dealer_sign/oos_summary.csv`, sample "all test days". % = change in mean loss versus the
baseline (negative is better); p = one-sided DM p that the model beats the baseline. (a): 452 days, QLIKE;
(b): 389 days, squared error.

| Model | (a) signed | (b) signed | (a) free | (b) free |
|---|---|---|---|---|
| US convention | +0.14% (p = 0.77) | +1.01% (p = 0.68) | +0.48% (p = 0.84) | +1.01% (p = 0.68) |
| Pro | +0.68% (p = 0.68) | -1.57% (p = 0.029) | +0.68% (p = 0.68) | **-5.22% (p = 0.0027)** |
| Client mirror | +1.74% (p = 0.90) | -0.06% (p = 0.47) | +1.74% (p = 0.90) | +0.07% (p = 0.51) |
| est_shares | +0.14% (p = 0.77) | +1.01% (p = 0.68) | +0.48% (p = 0.84) | +1.01% (p = 0.68) |
| est_participants | +0.98% (p = 0.88) | -2.33% (p = 0.25) | +1.59% (p = 0.84) | -2.33% (p = 0.25) |
| est_participants, frozen | +0.39% (p = 0.74) | +0.19% (p = 0.52) | +1.73% (p = 0.90) | +0.19% (p = 0.52) |
| Kalman, shares | | | -3.53% (p = 0.099) | +3.26% (p = 0.84) |
| Kalman, participants | | | +0.45% (p = 0.54) | -0.40% (p = 0.45) |

Sign of the theory-signed term across the 22 refits (`results/dealer_sign/oos_signed_coef_share.csv`): in
(a) Pro and Client mirror had the theory sign at 22 of 22 refits, while US, est_shares and est_participants
had the wrong sign at 22 of 22 (so in the signed template their (a) term was always set to 0). In (b) it is
the reverse: US and both estimates had the theory sign at 22 of 22, Pro and Client mirror the wrong sign at
22 of 22. So the signed Pro gain in (b) (-1.57%) comes from the free GEX level term alone, and the free Pro
gain (-5.22%) comes from an interaction with the anti-theory sign.

est_shares chose 315 degrees (the US convention) at every refit, so its forecasts are identical to the US
rule's.

### 4. Diebold-Mariano: estimated sign against the fixed rules (REAL DATA, primary tests)

Source: `results/dealer_sign/oos_dm.csv`, signed template, all test days. % = mean loss of the estimate
versus the rule (negative = estimate better); p = one-sided DM p that the estimate is better; Holm over the
10 tests that exist.

| Outcome | Estimate | vs US convention | vs Pro | vs Client mirror |
|---|---|---|---|---|
| (a) | est_participants | +0.84% (p = 0.83) | +0.29% (p = 0.59) | -0.75% (p = 0.30) |
| (a) | est_shares | identical forecasts | -0.54% (p = 0.35) | -1.57% (p = 0.12) |
| (b) | est_participants | -3.30% (p = 0.137) | -0.77% (p = 0.41) | -2.27% (p = 0.22) |
| (b) | est_shares | identical forecasts | +2.62% (p = 0.87) | +1.07% (p = 0.71) |

All 10 Holm-adjusted p-values are 1.0. Outside the primary set, in the free template the participant
estimate beats Client mirror in (b) by 2.4% (unadjusted p = 0.017) but loses to free Pro by 3.1%
(two-sided p = 0.35).

### 5. Sign agreement (REAL DATA)

Source: `results/dealer_sign/sign_agreement.csv` (quarterly rows are in the same file).

| Period | Estimated GEX negative | Same sign as US | Same sign as Pro | Same sign as Client mirror |
|---|---|---|---|---|
| 2023 to 2024 (direction 272 from that period) | 93.1% | 13.2% | 69.6% | 0.0% |
| test, expanding refits | 93.8% | 19.7% | 15.9% | 5.3% |

The shares estimate is the US rule (100% agreement in both periods).

### 6. Weight stability (REAL DATA)

Sources: `results/dealer_sign/weights_path_expanding.csv`, `weights_path_rolling.csv`, `kalman_params.csv`,
`kalman_summary.csv`; figure `results/dealer_sign/figures/dealer_weights_over_time.png`.

- Expanding window, participants (joint): 272 degrees at the first refit (2025-01-01), 260 to 264 until
  April 2025, 207 at 2025-05-08, then 189 to 201 at every later refit (190 at the last). The width of the
  LR set shrank from 135 to 40 degrees as data accumulated. The (a)-only direction stayed at 9 to 51 degrees
  throughout, the (b)-only direction equals the joint one.
- Rolling 250-day window, participants: 15 to 321 degrees in 2024 windows, 163 to 195 degrees from the
  window ending 2025-05-06 on, with bootstrap bands 170 to 280 degrees wide.
- Rolling window, shares: US side in every window ending from February to November 2024, mixed in windows
  ending from December 2024 to March 2026, reverse-US side in every window ending from 2026-04-13 on (94% to 98.5% of draws on the reverse-US side in
  the last five windows).
- Kalman model: the drift test rejects q = 0 in (a) (shares p = 0.0026, participants p = 0.0048) and for
  shares in (b) (p = 0.043), not for participants in (b) (p = 0.26). The filtered loadings are rarely
  distinguishable from zero: the Wald test of "no effect" passes its 90% critical value on 1.1% of test days
  for participants in (a) and 15.2% in (b).

### 7. Placebo (REAL DATA, shuffled outcomes)

Source: `results/dealer_sign/placebo_summary.csv` (draws in `placebo_draws.csv`, `placebo_kalman.csv`).
p = share of shuffles at least as extreme as the real value.

| Statistic | Real | Placebo median | Placebo 95th pct | p |
|---|---|---|---|---|
| joint LR, participants | 21.28 | 5.03 | 16.12 | 0.005 |
| LR (b), participants | 21.28 | 4.19 | 14.63 | 0.005 |
| LR (a), participants | 3.75 | 1.41 | 5.29 | 0.144 |
| joint LR, shares | 7.98 | 2.53 | 11.69 | 0.139 |
| OOS (b) est_participants vs baseline | -2.33% | +0.97% | +4.28% | 0.030 |
| OOS (a) est_participants vs baseline | +0.98% | +0.27% | +1.58% | 0.83 |
| Kalman drift test p, (a) participants | 0.0048 | 1.0 | 1.0 | 0.048 (20 shuffles) |
| Kalman drift test p, (b) participants | 0.26 | 0.14 | 0.52 | 0.67 (20 shuffles) |

The (b) direction and its out-of-sample gain survive the placebo; nothing in (a) does. Note that 0.048 is
the smallest p possible with 20 shuffles. The Kalman drift test in (b) is badly calibrated: 45% (participants)
and 40% (shares) of placebo fits reject q = 0 at 5%, so the (b) drift p-values should not be read as evidence.

### 8. SYNTHETIC validation (known weights, real regressors)

Source: `results/dealer_sign/synthetic_validation.csv` (draws in `synthetic_validation_draws.csv`).
200 draws per row; true angle drawn at random. Detection = LR above the 95% value simulated with no effect.

| Family | Signal-to-noise | Detection | Correct side | Median error (degrees) | 90% LR set coverage |
|---|---|---|---|---|---|
| participants | 0 (no effect) | 0.050 | 0.460 | 92.7 | 0.67 |
| participants | 0.05 | 0.355 | 0.940 | 22.2 | 0.88 |
| participants | 0.10 | 0.950 | 1.000 | 9.0 | 0.87 |
| participants | 0.20 | 1.000 | 1.000 | 4.5 | 0.91 |
| shares | 0 (no effect) | 0.050 | 0.605 | | 0.945 |
| shares | 0.05 | 0.555 | 0.985 | | 1.000 |
| shares | 0.10 | 0.980 | 1.000 | | 1.000 |
| shares | 0.20 | 1.000 | 1.000 | | 1.000 |

The 0.605 correct-side rate with no effect is chance: five more seeds give 0.455 to 0.500 (standard error
0.035; `synthetic_validation_null_seeds.csv`). The unit tests in `tests/test_dealer_sign.py` check the same
things on fully synthetic data: known participant angles (30, 150, 250 degrees) recovered within 15 degrees,
the side only for scale-free shares, wide bootstrap bands with no effect, a Kalman filter that tracks weights
drifting from 0 to 90 degrees, and an out-of-sample gain only when an effect exists.

---

## Robustness (REAL DATA)

Sources: `results/dealer_sign/direction_estimates.csv` (controls column) and
`results/dealer_sign/robustness_oos.csv`.

| Variant | Participant joint direction (full sample), 90% LR set | (a) direction, Wald p | OOS (b) est_participants vs baseline |
|---|---|---|---|
| base (auction days out of (b)) | 190, 178 to 217 | 21, p = 0.18 | -2.33% (p = 0.25) |
| auction days out of both outcomes | 190, 178 to 217 | 23, p = 0.11 | -2.33% (p = 0.25) |
| auction days kept in (b) as their own regime | 190, 178 to 215 | 21, p = 0.18 | -1.34% (p = 0.31) |
| expiry-regime and days-to-expiry controls | 193, 182 to 216 | 12, p = 0.034 | -1.95% (p = 0.22) |
| open-interest and gross-level controls | 191, 179 to 215 | 83, p = 0.70 (all angles) | -3.05% (p = 0.14) |
| old R3 scaling (comparison only) | 217, 186 to 266 | 3, p = 0.34 | -0.10% (p = 0.49) |

- The (b) "reverse Pro" direction holds in every variant (190 to 193 degrees with the scale-free measure).
- The (a) "Pro-like" direction depends on controls: with expiry-regime controls it strengthens (12 degrees,
  p = 0.034); with open-interest and gross-level controls it disappears (83 degrees, every angle in the LR
  set).
- In the 2023 to 2024 sample with open-interest and gross-level controls the (b) signal is gone as well
  (263 degrees, Wald p = 0.185), so the (b) result rests mainly on 2025 to 2026.
- No variant makes any DM comparison of an estimate against a fixed rule significant: the smallest
  one-sided p over all variants and both estimates is 0.101 (est_shares against Client mirror in (a),
  open-interest and gross-level controls).
- Shares: the joint side stays US (315) in every scale-free variant. Under the old R3 scaling the shares
  family seems to reveal a 2-D direction (52 degrees, (a) Wald p = 0.013), an artifact of the expiry-eve
  swing in the scaling: the mean of gross GEX / 20-day median, measured at the close before the outcome
  day, is 1.71 when the outcome day is an expiry day, then 1.24, 0.99, 0.74 and 0.52 when it is one to four
  trading days before expiry (`results/dealer_sign_stdout.txt`).

---

## Limitations

- **The answer is only as good as the maintained hypothesis.** If hedging theory is wrong for Nifty, or if
  something other than dealer hedging (for example retail flows that are correlated with Pro positions)
  drives the momentum pattern, the revealed sign describes that force, not dealers.
- The two outcomes disagree, so no single sign satisfies the hypothesis. Either the hypothesis fails for one
  outcome, or the Pro and Client-mirror series are poor proxies for dealer positions (their scale-free
  versions are correlated at 0.569 and vary little: standard deviations 0.021 and 0.025).
- The shares family can only tell the side of w_call - w_put once the scale problem is removed. A
  strike-level or moneyness-level split would be needed to say more.
- Participant OI is published once a day for the whole index-option book, not per strike or expiry; the
  model assumes each category holds the same mix of strikes as the market.
- About 450 test days and signal-to-noise near 0.035 to 0.07 in (a) give little power: in the SYNTHETIC
  check detection is 35.5% at signal-to-noise 0.05 (and 95% at 0.10).
- The LR set over-covers with no effect and is not a calibrated confidence set; bootstrap and placebo are
  the better guides. The Kalman drift test in (b) is miscalibrated (section 7).
- The closing-auction period has only 63 days, too few to estimate a separate direction.

---

## Files and how to rerun

```
OMP_NUM_THREADS=1 python experiments/dealer_sign.py      # about 2 minutes (110.8 s on the last run)
OMP_NUM_THREADS=1 python -m pytest -q tests/test_dealer_sign.py
```

- `experiments/dealer_sign.py`: the whole study (`python experiments/dealer_sign.py --help` for options;
  defaults: 200 bootstrap draws, 200 shuffles, 20 Kalman shuffles, 200 synthetic draws, seed 82).
- `tests/test_dealer_sign.py`: 18 synthetic tests with known weights.
- `results/dealer_sign/`: `frame.csv` (the analysis panel), the CSV tables named above, `run_info.json`,
  and `figures/dealer_weights_over_time.png`, `figures/dealer_weights_kalman.png`,
  `figures/lr_profiles.png`.
- `results/dealer_sign_stdout.txt`: the printed log of the last run.
