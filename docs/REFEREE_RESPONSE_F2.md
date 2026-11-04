# F2 under the referee's questions (REAL DATA, post hoc)

**Everything here is REAL DATA**: NSE participant-wise open interest, the v2 daily GEX panel built from NSE
option bhavcopies (`results/pipeline_nse_v2/`), and the validated 1-minute Nifty 50 bars, 2 January 2023 to
2 November 2026. Nothing is simulated except the unit tests.

**Everything here is post hoc.** Every analysis was chosen after reading the referee report of 3 November 2026
(`REFEREE_REPORT.md`, points M2.2, M2.4, M2.5, M3, M10 and M13) and after every F2 result in the repository had
been seen. None of it is pre-registered. The script reports 598 p-values, and Section 8's script (referee
point M1(a), added later) 381 more; none of the 979 is adjusted for that count, so a single p-value near 0.02
here is weak evidence.

    OMP_NUM_THREADS=1 python experiments/referee_f2.py            # about 3 minutes on one core
    OMP_NUM_THREADS=1 python -m pytest -q tests/test_referee_f2.py

* Script: `experiments/referee_f2.py`. Unit tests (synthetic data only): `tests/test_referee_f2.py`.
  Section 8: `experiments/referee_f2_positions.py` and `tests/test_referee_f2_positions.py`, results in
  `results/referee_f2/item8_*` and the log `results/referee_f2_positions_stdout.txt`.
* Results: `results/referee_f2/` (one CSV per item, named below, plus `input_checks.csv` and `run_info.json`
  with the seed, the number of draws and the input file hashes) and the printed log
  `results/referee_f2_stdout.txt`.
* Bootstrap and permutation p-values come from random draws, and each item draws its own, so the same
  regression can show slightly different simulated p-values in two files (for example the Pro wild month p is
  0.0010 in `item3_f2_inference.csv` and 0.0014 in `item4_other_rules.csv`). Formula-based p-values agree
  exactly.
* SEBI dates and their sources: `data/external/sebi_phases.csv` (new, same layout as
  `data/external/sensex_expiry_calendar.csv`).

## F2 in one paragraph

For each trading day, r_rod is Nifty's return from the previous day's 15:29 bar close to today's 14:59 bar
close (the "rest of the day", overnight gap included) and r_last the return from the 14:59 bar close to the
15:29 bar close (the "last 30 minutes"), both in basis points (bp). F2 regresses r_last on r_rod, on a 0/1
marker N for "GEX signed by NSE's Pro positions was negative at the previous evening's close", and on
r_rod x N. The coefficient on r_rod x N (the "F2 coefficient" below) says how much more the last 30 minutes
reverse the day's move after a negative-Pro evening. On the v2 panels it is **-0.0680** (Newey-West p 0.0002)
on 932 days, 466 of them after a negative-Pro evening (`item3_f2_inference.csv`). In plain words: after a
negative-Pro evening, a 100 bp rise over the day is followed by about 6.8 bp more fall in the last 30 minutes
than after other evenings.

Why 932 days and not 929: the published R5 code divided GEX by a 20-day median that needs 5 evenings, so the
first 3 outcome days of January 2023 had no GEX. The sign does not depend on that scale, so here the sign is
taken from the unscaled GEX and those 3 days stay in. That is the whole difference from the published
-0.0681 and -0.0682. The script checks its inputs against the earlier studies (`input_checks.csv`): its
r_rod and r_last equal `experiments/momentum_gex.py` exactly, and the Pro rule rebuilt from the chain and the
participant file matches the panel to 2.5e-17 of gross GEX.

## Words used here

* **Episode.** A run of consecutive evenings with the same Pro sign (all negative or all non-negative).
* **Standard error, p-value.** The standard error measures how much a coefficient would move by chance; the
  p-value is the chance of a result at least this far from zero if the true effect were zero.
* **Newey-West (HAC).** A standard error that allows the errors of nearby days (here up to 5 days apart) to be
  related. It is what the paper used. It is too optimistic when the regressor stays the same for weeks.
* **Driscoll-Kraay.** Built for panels (many units observed on the same days). With one time series, as here,
  it is Newey-West on the regression's scores with a lag that grows with the sample, floor(4 (T/100)^(2/9)),
  which is 6 for 932 days and 5 for about 450 days. So it adds little; we report it because it was asked for.
* **Newey-West with 60 lags and fixed-b p-values.** Allows errors up to 60 trading days apart to be related.
  With such a long window the usual normal p-value is too small, so the p-value is taken from a simulated
  "fixed-b" distribution (Kiefer and Vogelsang 2005).
* **Clustered standard errors.** Days are grouped (by calendar month, by episode, or by calendar quarter) and
  any correlation inside a group is allowed. "CR1" is the usual formula, with a t distribution with
  (groups minus 1) degrees of freedom.
* **Wild cluster bootstrap.** The CR1 p-value is unreliable with few groups or with groups of very different
  size. The wild cluster bootstrap (restricted, Webb weights; Cameron, Gelbach and Miller 2008, MacKinnon and
  Webb 2017) imposes "no effect", flips the residuals of whole groups at random 4,999 times, and asks how often
  the fake t-statistic is as large as the real one. It is the more trustworthy cluster p-value.
* **Block permutation.** The 0/1 series N is cut into blocks of 60 (or 120) consecutive days and the blocks are
  put in a random order (after a random rotation), 4,999 times. Each block keeps its runs, so the fake series
  is as persistent as the real one, but its timing no longer lines up with the returns. The p-value is the
  share of fake series that give an F2 statistic at least as extreme as the real one. We report it for two
  statistics: the Newey-West t ("_t") and the raw coefficient ("_coef"). They are both valid; when they
  disagree, the larger p-value is the cautious reading.
* **Within-block shuffle.** The scheme of `docs/FALSIFICATION.md` T1 (shuffle days inside consecutive blocks),
  here with 60 and 120 days. It keeps the slow level of N but not its runs, so it is reported only for
  comparison.
* **Moving-block bootstrap.** Rebuild the sample 9,999 times from randomly chosen stretches of 60 consecutive
  days and re-estimate; the spread of the estimates gives an interval.
* **Autocorrelation, effective sample size.** The autocorrelation at lag k is the correlation of the series
  with itself k days earlier. A persistent series carries less information than its length suggests. Two
  measures of how many independent observations it is worth: n (1 - r1) / (1 + r1), with r1 the lag-1
  autocorrelation, and n / tau, with tau the integrated autocorrelation time (1 + 2 x the sum of the
  autocorrelations, summed up to Sokal's automatic window). The second counts slow drifts, the first does not.

---

## 1. The cleanest hedging test: unconditional end-of-day momentum (M2.5)

**What the referee asked.** Under "dealers are short all options" (the `short_all` rule of
`src/niftygex/gex.py`: net GEX is minus gross GEX on every evening) hedging theory predicts end-of-day
momentum, that is, a positive slope b of r_last on r_rod, with no participant proxy at all. Present that test
first.

**What we did.** Re-estimated b on the v2 panel with Newey-West (5 lags), Driscoll-Kraay, Newey-West with 60
lags (fixed-b), month clusters (CR1 and wild bootstrap) and a 60-day moving-block bootstrap. We also ran the
"size" version of the same theory: if dealers are short all options, momentum should be stronger when there
is more gamma, so the slope should rise with gross GEX (at the previous close, relative to its 20-day median).

**Result (REAL DATA, `results/referee_f2/item1_unconditional_slope.csv`).** b in bp per bp; theory predicts
b > 0. "p, b > 0" is the one-sided p-value for momentum.

| Sample | Days | b | Newey-West p | Month clusters: CR1 p / wild p | 60-day block bootstrap: 95% interval, p | p, b > 0 |
|---|---|---|---|---|---|---|
| full | 932 | -0.0294 | 0.015 | 0.010 / 0.023 | -0.048 to -0.005, 0.0094 | 0.99 |
| before the closing auction | 869 | -0.0303 | 0.017 | 0.011 / 0.025 | -0.048 to -0.002, 0.018 | 0.99 |
| 2023 to 2024 | 480 | -0.0412 | 0.020 | 0.020 / 0.051 | -0.065 to 0.006, 0.053 | 0.99 |
| 2025 onward | 452 | -0.0193 | 0.20 | 0.17 / 0.25 | -0.045 to 0.003, 0.115 | 0.90 |
| expiry days | 198 | -0.0374 | 0.019 | 0.033 / 0.036 | -0.075 to -0.026, 0.024 | 0.99 |
| non-expiry days | 734 | -0.0270 | 0.062 | 0.058 / 0.11 | -0.046 to 0.005, 0.060 | 0.97 |

Driscoll-Kraay and the 60-lag fixed-b p-values for the full sample are 0.015 and 0.018. The size version: the
slope changes by +0.0090 per standard deviation of gross GEX (Newey-West p 0.36, wild month p 0.39; one-sided
p for the predicted positive sign 0.18).

**What it changes.** This is now the first test of hedging theory, and it fails: in no sample is the slope
positive (the one-sided p-value for momentum is 0.90 or more everywhere), and more gamma does not bring more
momentum. In the full sample the slope is significantly negative (a mild reversal) under every method, so the
full-sample statement "Nifty reverses into the close, about 2.9 bp per 100 bp" stands. The reversal is not
significant in 2025 onward (p 0.115 to 0.25) or on non-expiry days (p 0.06 to 0.11), so it should be described
as a full-sample and expiry-day pattern, not as a property of every period. The plain reading is that the
"dealers short everything and hedge into the close" story does not describe Nifty in 2023 to 2026.

---

## 2. What the Pro sign is (M2.2, M3)

**What the referee asked.** Is the negative-Pro state a daily state or a slow trend? Tabulate it by year and
quarter, give run lengths, its autocorrelation and the effective number of independent episodes, and do the
same for FII, DII and Client.

**What we did.** For each evening (the publish date of the GEX used the next day), net GEX = w_call x C +
w_put x P, with C and P the gamma of the front Nifty chain's calls and puts and (w_call, w_put) the
category's net index-option position as a share of all open contracts (from `data/raw/participant_oi`). We
did this for Pro, FII, DII, Client, Client mirror (minus Client), FII plus Pro, and the US convention. We
also tabulated each category's raw positions.

**Result (REAL DATA).** Negative-GEX evenings by year (`item2_by_year.csv`):

| Year | Evenings | Pro | FII | DII | Client | Client mirror | FII plus Pro | US convention |
|---|---|---|---|---|---|---|---|---|
| 2023 | 245 | 232 (94.7%) | 0 | 0 | 239 | 6 | 14 | 16 |
| 2024 | 245 | 145 (59.2%) | 21 | 0 | 218 | 27 | 34 | 20 |
| 2025 | 247 | 58 (23.5%) | 6 | 0 | 241 | 6 | 7 | 34 |
| 2026 (to 2 Nov) | 205 | 39 (19.0%) | 0 | 0 | 205 | 0 | 0 | 37 |

Negative Pro evenings by quarter (`item2_by_quarter.csv`): 61 of 62 in 2023Q1 and 55 to 58 in each later
quarter of 2023; 26 to 46 per quarter in 2024; then 30 of 61 (2025Q1), 17 of 61 (2025Q2), 1 of 64 (2025Q3),
10 of 61 (2025Q4), 30 of 59 (2026Q1), 1 of 60 (2026Q2), 7 of 65 (2026Q3) and 1 of 21 (2026Q4).

By SEBI phase (`item2_by_segment.csv`; dates in Section 5.4): the Pro share is 76.4% before 1 February 2025
(392 of 513 evenings), 39.5% from 1 February to 31 March 2025 (15 of 38), 27.0% from 1 April to 2 July 2025
(17 of 63) and 15.2% from 3 July 2025 (50 of 328). Client mirror and FII are never negative from 3 July 2025.

Runs and persistence of the Pro indicator (`item2_summary.csv`, `item2_runs.csv`, `item2_acf.csv`):

| Period | Evenings | Negative | Negative runs | of 5+ evenings | of 10+ | Longest | Share of negatives in the 3 longest runs | Lag-1 autocorrelation | n (1 - r1)/(1 + r1) | n / tau |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 942 | 474 | 86 | 20 | 14 | 60 | 33.3% | 0.637 | 209 | 9.5 |
| 2023 to 2024 | 490 | 377 | 54 | 16 | 11 | 60 | 41.9% | 0.390 | 215 | 19.1 |
| 2025 onward | 452 | 97 | 33 | 4 | 3 | 20 | 49.5% | 0.572 | 123 | 31.1 |

Over all evenings the sign changes 171 times, the median negative run lasts 2 evenings, and the
autocorrelation is still 0.410 at 20 days and 0.365 at 60 days. The longest negative runs are 6 January to
5 April 2023 (60 evenings), 9 October 2023 to 3 January 2024 (59), 10 April to 5 June 2023 (39), 7 June to
6 July 2023 (21), 24 March to 24 April 2025 (20), and 9 to 30 March 2026 (15). (The 0.80 in
`docs/FALSIFICATION.md` is the lag-1 autocorrelation of the continuous Pro share; the 0/1 indicator's is 0.637.)

Other categories over all evenings: FII is negative on 27 evenings (15 runs, none longer than 5), DII never
(DII holds index puts and almost no calls), Client on 903 (so Client mirror on 39, in 24 runs), FII plus Pro
on 55.

Raw positions (`item2_positions.csv`, share of trading days): Pro is net short both index calls and puts on
63.7% of days in 2023, 25.3% in 2024, 10.1% in 2025 and 3.9% in 2026, and net long both on 0.8%, 11.4%, 49.4%
and 51.2%. FII is net long both on 99.6%, 84.5%, 64.0% and 6.8% (net short calls and long puts on 93.2% of
2026 days). Client is net short both on 61.6%, 68.2%, 66.0% and 11.7% (net long calls and short puts on 88.3%
of 2026 days). This confirms the referee's tabulation (the referee's "100%" for FII in 2023 is 99.6% here).

**What it changes.** The negative-Pro state is a slow, regime-like series, not a daily state. It falls from 95%
of evenings in 2023 to about a fifth from 2025, in steps that line up with SEBI's 2025 phases, and after 2024
it comes mostly in two quarters (2025Q1 and 2026Q1, 30 evenings each). Counted honestly, the full sample holds
something between about 10 independent observations of the state (when its slow drift is counted) and about
200 (when only day-to-day persistence is counted); 2025 onward has 4 negative runs of 5 or more evenings.
The paper should describe N as "Pro's net index-option position state", report this table, and stop treating
it as a daily dealer-gamma signal.

---

## 3. Inference fit for a persistent state (M3)

**What the referee asked.** Newey-West with 5 lags is not adequate for a state with runs of 20 days; report
clusters by month and by episode, Driscoll-Kraay, and block permutations with 60- and 120-day blocks, on the
full sample, 2023 to 2024 and 2025 onward.

**What we did.** All of that, plus quarter clusters (16 quarters in the full sample) as the coarsest grouping,
wild cluster bootstraps for every clustering, and both the t and the coefficient versions of the permutation.

**Result (REAL DATA, `results/referee_f2/item3_f2_inference.csv`).** F2 coefficient (Pro rule, binary
specification) and two-sided p-values.

| Method | Full sample | 2023 to 2024 | 2025 onward |
|---|---|---|---|
| F2 coefficient | -0.0680 | -0.0644 | -0.0739 |
| days (negative-Pro days; negative episodes) | 932 (466; 85) | 480 (368; 53) | 452 (98; 33) |
| Newey-West, 5 lags | 0.0002 | 0.020 | 0.012 |
| Driscoll-Kraay | 0.0002 | 0.020 | 0.012 |
| Newey-West, 60 lags, fixed-b | 0.0007 | 0.056 | 0.0034 |
| month clusters, CR1 / wild | 0.0003 / 0.0010 | 0.030 / 0.052 | 0.0030 / 0.0048 |
| episode clusters, CR1 / wild | 0.0004 / 0.0066 | 0.043 / 0.095 | 0.0009 / 0.0030 |
| quarter clusters, CR1 / wild | 0.0035 / 0.012 | 0.12 / 0.18 | 0.0024 / 0.023 |
| block permutation, 60 days: t / coef | 0.0010 / 0.0042 | 0.10 / 0.14 | 0.023 / 0.038 |
| block permutation, 120 days: t / coef | 0.0010 / 0.0020 | 0.099 / 0.14 | 0.018 / 0.029 |
| within-block shuffle, 60 days: t / coef | 0.0020 / 0.0042 | 0.053 / 0.090 | 0.027 / 0.055 |

There are 170 episodes (of both signs) in the full sample, 105 in 2023 to 2024 and 66 in 2025 onward; 47, 24
and 23 months; 16, 8 and 8 quarters. A 120-day block permutation of 2025 onward has only 4 blocks, so its
p-value is coarse.

With the VIX and move-size controls of R5 (same file, `spec = controls`): full sample -0.0496 (p between
0.0025 and 0.035 over all methods; the largest is wild quarter), 2023 to 2024 -0.0404 (p 0.12 to 0.30 under
every method), 2025 onward -0.0702 (p 0.030 or below under every method).

**What it changes.** In the full sample and in 2025 onward, F2 survives every method that respects the
persistence of the state: the largest p-value is 0.012 (full sample, wild quarter clusters) and 0.038 (2025
onward, 60-day block permutation of the coefficient). The referee's worry that the 2025-onward estimate rests
on a handful of episodes is right about the count (33 negative episodes, 4 of them 5 evenings or longer) but
the estimate holds when whole 60- or 120-day blocks are moved. **In 2023 to 2024 F2 does not survive**: under
episode or quarter clusters and block permutations p is 0.095 to 0.18, and with the R5 controls it is gone
(p 0.12 to 0.30). The claim "F2 holds in both halves of the sample" must become "F2 is robust in the full
sample and in 2025 onward; in 2023 to 2024, when 95% of 2023 evenings were negative, it is not distinguishable
from noise once the persistence of the state is respected".

---

## 4. Other categories as dealer proxies (M2.4)

**What the referee asked.** Report F2 with FII, FII plus Pro, DII and Client mirror as the dealer, and with
`short_all`.

**What we did.** The same F2 regression with N taken from each rule, in the binary and the controls
specification, on the three samples, with Newey-West, month and episode clusters (CR1 and wild) and a 60-day
block permutation.

**Result (REAL DATA, `results/referee_f2/item4_other_rules.csv`).** Full sample, binary specification unless
stated.

| Rule | Negative days (episodes) | F2 coefficient | Newey-West p | wild month p | wild episode p | 60-day block permutation p (coef) | Coefficient with VIX and size controls (Newey-West p) |
|---|---|---|---|---|---|---|---|
| Pro | 466 (85) | -0.0680 | 0.0002 | 0.0014 | 0.0066 | 0.0038 | -0.0496 (0.0029) |
| FII | 27 (15) | -0.0660 | 0.0001 | 0.16 | 0.18 | 0.30 | +0.0287 (0.45) |
| FII plus Pro | 55 (36) | -0.0670 | 0.0001 | 0.13 | 0.14 | 0.15 | -0.0004 (0.99) |
| DII | 0 | not estimable: DII GEX is never negative | | | | | |
| Client mirror | 39 (24) | -0.0731 | 0.00003 | 0.064 | 0.086 | 0.19 | -0.0007 (0.99) |
| Client | 893 (25) | +0.0731 | 0.00003 | 0.069 | 0.093 | 0.18 | +0.0007 (0.99) |
| US convention | 107 (77) | +0.0129 | 0.59 | 0.51 | 0.61 | 0.75 | -0.0008 (0.97) |
| short_all | 932 (1) | not identified: N = 1 on every evening, so F2 reduces to the unconditional slope of Section 1 | | | | | |

Client is the exact mirror of Client mirror (its N is 1 minus theirs), so its coefficient has the opposite
sign and the same p-values up to simulation noise. In 2025 onward FII and Client mirror are negative on only
6 evenings in 3 episodes; their large coefficients there (-0.0921 and -0.1042) have wild episode p 0.28 and
0.26 and permutation p 0.44 and 0.39.

**What it changes.** F2 is specific to the Pro rule. FII, FII plus Pro and Client mirror give binary
coefficients of the same size, but on 27 to 55 days, which is why their Newey-West p-values look strong; under
wild cluster bootstraps and the block permutation of the coefficient they are not significant (p 0.06 to
0.30). The block permutation of the Newey-West t disagrees (p 0.009 to 0.020); by the rule above we take the
larger p-value. The VIX and size controls remove them completely (coefficients -0.0007 to +0.0287, p 0.45 or
more). In those rules the "negative" evenings are a few volatile days, as R5 already
said for Client mirror. DII never gives a negative sign. So the category choice matters, no category other than
Pro supports F2, and the paper can no longer say that a participant-based dealer sign in general produces it.
This supports the referee's reading in M1: F2 is a fact about Pro's net position, not about who hedges.

---

## 5. Alternative explanations (M3)

All rows: Pro rule, binary specification, full sample unless stated (`results/referee_f2/item5_alternatives.csv`).
The last two columns give the most cautious cluster p-value (wild bootstrap by episode) and the 60-day block
permutation of the coefficient.

| Case | F2 coefficient | Newey-West p | wild month p | wild episode p | block permutation p |
|---|---|---|---|---|---|
| baseline, 14:59 to 15:29 | -0.0680 | 0.0002 | 0.0014 | 0.0066 | 0.0024 |
| 14:59 to 15:15 only | -0.0395 | 0.014 | 0.070 | 0.069 | 0.053 |
| 15:15 to 15:29 only | -0.0285 | 0.0019 | 0.0048 | 0.0038 | 0.0078 |
| first bar dropped: 15:00 to 15:29 | -0.0646 | 0.0001 | 0.0008 | 0.0012 | 0.0012 |
| first bar only: 14:59 to 15:00 | -0.0034 | 0.61 | 0.58 | 0.62 | 0.60 |
| baseline, before the auction (869 days) | -0.0719 | 0.0001 | 0.0004 | 0.0054 | 0.0026 |
| 14:59 to 15:15, before the auction | -0.0453 | 0.0072 | 0.044 | 0.044 | 0.027 |
| 15:15 to 15:29, before the auction | -0.0266 | 0.0022 | 0.0074 | 0.0064 | 0.013 |
| year-quarter fixed effects (levels) | -0.0690 | 0.0001 | 0.0012 | 0.0046 | 0.0018 |
| year-month fixed effects (levels) | -0.0701 | 0.0001 | 0.0008 | 0.0040 | 0.0032 |
| quarter-of-year levels and slopes | -0.0679 | 0.0002 | 0.0032 | 0.0062 | 0.0060 |
| month-of-year levels and slopes | -0.0712 | 0.0001 | 0.0008 | 0.0028 | 0.0062 |
| year-quarter levels and slopes (only within-quarter variation is used) | -0.0790 | 0.00003 | 0.0006 | 0.0002 | 0.0020 |
| outside January to March | -0.0607 | 0.011 | 0.062 | 0.12 | 0.060 |
| inside January to March | -0.0890 | 0.0043 | 0.0060 | 0.0026 | 0.020 |
| difference, inside minus outside | -0.0283 | 0.47 | 0.48 | 0.52 | 0.60 |
| SEBI phases: levels and slopes | -0.0687 | 0.0005 | 0.0014 | 0.0006 | 0.0030 |
| SEBI phases plus expiry-regime levels and slopes | -0.0711 | 0.0002 | 0.0004 | 0.0004 | 0.0046 |
| Sensex expiry day level and slope | -0.0688 | 0.0002 | 0.0008 | 0.0056 | 0.0038 |
| Sensex expiry days dropped (751 days) | -0.0662 | 0.0032 | 0.020 | 0.041 | 0.028 |
| all controls together | -0.0530 | 0.0020 | 0.0008 | 0.0042 | 0.030 |

"All controls together" adds the R5 VIX and move-size controls, the SEBI phase and expiry-regime levels and
slopes, quarter-of-year levels and slopes, and the Sensex level and slope (year-quarter levels are left out
because the 1 April 2025 phase starts a quarter and would be collinear with them).

### 5.1 Broker auto square-off and the closing-price window (M3.1, M3.2)

**What we did.** Split the last 30 minutes at the 15:15 bar close: 14:59 to 15:15 (16 one-minute bars) and
15:15 to 15:29 (14 bars). Broker square-off of intraday positions runs at roughly 15:15 to 15:25, so it would
put F2 in the second part. Before August 2026 the official close was the average of 15:00 to 15:30, so flows
benchmarked to the close would spread over both parts.

**Result.** F2 is in both parts: -0.0395 before 15:15 and -0.0285 after (they add up to -0.0680). The second
part is the more precisely estimated (p 0.004 to 0.008 under the robust methods, against 0.053 to 0.070 for
the first part in the full sample and 0.027 to 0.044 before the auction).

**What it changes.** Square-off cannot be the whole story, because more than half of the coefficient arrives
before 15:15, but the sharper second part is consistent with some role for it. The averaging window covers
both parts, so this split cannot separate F2 from close-benchmarked trading; a test of that needs order-flow
or broker data that we do not have. We can only say that F2 is spread over the last 30 minutes and is not
concentrated in the square-off window.

### 5.2 Stale prints (M3.6)

**Result.** The 14:59 to 15:00 bar carries none of F2 (-0.0034, p 0.6), and dropping it leaves -0.0646
(p 0.0008 to 0.0012 under the robust methods). Section 7 adds that the last minute of r_rod and the first
minute of r_last are not negatively correlated (0.023, p 0.49), so there is no bounce at the junction.

**What it changes.** Stale or bouncing prints at 14:59 do not explain F2. The fall in the trading rule's gross
edge when entering one minute later (`results/f2_economics/metrics.csv`, rule D) is about the rule's payoff,
which also contains the unconditional part of the reversal; the F2 coefficient itself is not in the first bar.

### 5.3 Seasonality (M3.3)

**Result.** Month or quarter fixed effects, and seasonal slopes by quarter or month of the year, leave F2
between -0.0679 and -0.0712. Using only variation inside each calendar quarter (each quarter gets its own
level and slope) gives -0.0790 (p 0.002 or below). Inside January to March the coefficient is -0.0890
(237 January-to-March days, 143 of them negative-Pro, columns `n_janmar` and `n_neg_janmar`); outside it is
-0.0607; the difference (-0.0283) is far from significant (p 0.46 to 0.60).

**What it changes.** F2 is not a January-to-March artefact and not a between-quarter (period-dummy)
artefact: it is at least as strong within quarters. One caution: outside January to March alone the evidence
is borderline under the robust methods (wild episode p 0.12, block permutation p 0.060).

### 5.4 SEBI's phased measures and the July 2025 order (M3.4)

Sources (`data/external/sebi_phases.csv`). No primary SEBI document could be fetched: `www.sebi.gov.in` (and
`www.scconline.com`) could not be resolved from this container, so every date is "reported", taken from
search-engine results for news, law-firm and broker summaries, as in `data/external/sensex_expiry_calendar.csv`.

| Date | Measure | Sources (not fetched) |
|---|---|---|
| 1 Feb 2025 | upfront collection of option premium from buyers; no calendar-spread margin benefit on expiry day (both under SEBI's circular of 1 Oct 2024, "Measures to strengthen Equity Index Derivatives framework for increased investor protection and market stability") | Business Standard (3 Oct 2024); Moneylife; independentdirectorsdatabank.in newsletter; KPMG BFSI newsletter (Nov 2024). One secondary outlet reportedly gives 5 Feb 2025 for the premium rule. |
| 1 Apr 2025 | intraday monitoring of position limits in index derivatives (same circular) | Business Standard (3 Oct 2024); Moneylife |
| 3 Jul 2025 | SEBI interim order in the matter of index manipulation by Jane Street Group | SEBI's order list (title and date only, from search results); Business Standard (4 Jul 2025); Moneylife |

**What we did.** Added step dummies for the three dates and their products with r_rod (the reversal slope may
change at each phase), alone and together with the expiry-regime steps of `daily_market.csv` (single weekly
Thursday expiry from 21 November 2024, Tuesday expiry from 1 September 2025, closing auction from 3 August 2026).

**Result.** -0.0687 and -0.0711, robust p 0.0004 to 0.0046.

**What it changes.** The regulatory breaks do not explain F2: it holds inside the phases. But Section 2 shows
that the Pro sign itself moved with the phases (76% negative before February 2025, 15% after 3 July 2025),
which supports the referee's view that N partly marks regimes.

### 5.5 Sensex expiry days (M3.5)

Sensex expiry days come from the reported weekday regimes in `data/external/sensex_expiry_calendar.csv` with
the holiday roll-back rule; they agree with `results/sensex_confound/sensex_expiry_days_used.csv` on every day
(181 of the 932 F2 days; `input_checks.csv`). The calendar is reported, not verified against BSE files.

**Result.** With a Sensex-expiry level and slope: -0.0688 (robust p 0.0008 to 0.0056). Dropping the 181
Sensex days: -0.0662 (robust p 0.020 to 0.041).

**What it changes.** Sensex expiry days do not explain F2. The Limitations sentence "F1, F2 and the pinning
tests were not re-estimated with it" can be updated for F2.

---

## 6. Are the two dealer-sign angles different? (M13)

**What the referee asked.** `docs/DEALER_SIGN.md` says outcome (a) (next-day variance) points to 21 degrees
(mostly Pro) and outcome (b) (end-of-day momentum) to 190 degrees (reverse Pro), 169 degrees apart. Since (a)
is weak (Wald p 0.18), test whether the two angles are equal with a joint bootstrap.

**What we did.** The angle of each outcome is the direction of minus its unrestricted loadings theta (the
script checks that this closed form gives the same 21 and 190 degrees as the grid search of
`experiments/dealer_sign.py`). We resampled the same blocks of calendar days for both outcomes (blocks of 10
days, as in DEALER_SIGN.md, and of 60 days), 4,999 times, and re-estimated both angles each time. Two tests:

* **angle bootstrap**: is the difference of the two angles far from 0 compared with its bootstrap spread?
* **minimum-distance test**: can one direction fit both outcomes' loadings, letting each outcome have its own
  non-negative strength (including zero)? The statistic J is compared with a chi-square with 1 degree of
  freedom (the usual reference) and with 2 (the right one when (a)'s strength is at zero).

**Result (REAL DATA, `results/referee_f2/item6_angle_equality.csv`; draws in
`item6_angle_bootstrap_draws.csv.gz`).**

| Sample | Blocks | Angle (a) | Angle (b) | Difference (a - b) | 95% interval of the difference | Angle bootstrap p | Draws with the two angles within 90 degrees | Minimum-distance J, p (chi-square 1 / 2) |
|---|---|---|---|---|---|---|---|---|
| full | 10 days | 21.4 | 190.2 | -168.8 | 97.6 to -59.1 (through 180) | 0.0018 | 9.4% | 3.23, 0.072 / 0.20 |
| full | 60 days | 21.4 | 190.2 | -168.8 | 90.7 to -62.0 (through 180) | 0.0036 | 11.0% | 2.54, 0.11 / 0.28 |
| 2023 to 2024 | 10 days | 29.2 | 271.7 | 117.5 | -28.9 to -91.0 (through 180, contains 0) | 0.15 | 37.5% | 1.98, 0.16 / 0.37 |
| 2023 to 2024 | 60 days | 29.2 | 271.7 | 117.5 | -16.6 to -106.1 (through 180, contains 0) | 0.10 | 36.4% | 2.52, 0.11 / 0.28 |

In the full sample the common direction that fits best is 189.9 degrees, the (b) direction, with (a)'s strength
set to zero; J is then exactly the Wald test that (a) has no gamma effect at all, using the bootstrap covariance
(p 0.20 with 10-day blocks, 0.28 with 60-day blocks).

**What it changes.** Taken at face value, the two angles differ (angle bootstrap p 0.002 to 0.004, and the
interval for the difference does not contain 0). But that comparison treats angle (a) as well measured, and it
is not: the data cannot reject that outcome (a) has no gamma effect, and once that is allowed one direction
(about 190 degrees, reverse Pro) fits both outcomes (p 0.07 to 0.28). In 2023 to 2024 the angles cannot be told
apart at all (p 0.10 to 0.15). So "signs estimated from the two outcomes are 169 degrees apart" is a statement
about point estimates; the paper should say "outcome (b) points to the reverse of the Pro sign; outcome (a)
does not identify a direction, so the disagreement between them is not established".

---

## 7. Microstructure noise in the last 30 minutes (M10, F2 side)

**What the referee asked.** The first-order autocorrelation of 1-minute Nifty returns in 14:59 to 15:29 on
expiry and other days, by period. A negative autocorrelation is the usual sign of noise (prices bouncing around
the true value), which inflates realized variance measured from 1-minute returns.

**What we did.** Pooled the 29 pairs of consecutive 1-minute returns in the window over all days of a group
(correlation of all pairs), with 95% intervals and p-values from a 1,999-draw day bootstrap; the same for the
rest of the day (09:15 to 14:59) as a reference; and the correlation across the 14:59 junction (the last minute
of r_rod against the first minute of r_last). The 63 closing-auction days are shown separately: their window
contains 13 frozen minutes and an auction jump.

**Result (REAL DATA, `results/referee_f2/item7_minute_autocorrelation.csv`, `item7_junction.csv`).**

| Days | Expiry: autocorrelation (95% interval) | Other days | Expiry minus other (p) | Noise share of 1-minute variance on expiry days (i.i.d. noise model) |
|---|---|---|---|---|
| all before the auction | -0.054 (-0.103 to -0.004) | +0.011 (-0.025 to 0.050) | -0.065 (0.034) | 10.8% |
| 2023 to 2024 | -0.034 (-0.104 to 0.037) | -0.007 | -0.027 (0.54) | 6.7% |
| 2025 onward, before the auction | -0.082 (-0.147 to -0.017) | +0.036 | -0.118 (0.0055) | 16.3% |
| regime multi-weekly | -0.051 | -0.007 | -0.044 (0.29) | 10.2% |
| regime single weekly, Thursday | -0.043 | +0.007 | -0.051 (0.50) | 8.6% |
| regime single weekly, Tuesday (before the auction) | -0.067 | +0.048 | -0.115 (0.038) | 13.5% |
| closing auction | -0.019 | +0.006 | -0.025 (0.24) | 3.8% |

Reference window, 09:15 to 14:59, before the auction: +0.020 on expiry days and +0.025 on other days
(difference p 0.52). Across the 14:59 junction the correlation is +0.023 (p 0.49; expiry days +0.118, p 0.11).
By Pro sign (before the auction), the last-30-minute autocorrelation on non-expiry days is +0.019 after
negative-Pro evenings and 0.000 after other evenings.

Under the i.i.d. noise model the share of noise in the variance of a 1-minute return is minus twice the
autocorrelation, and the log of 1-minute realized variance is overstated by minus the log of one minus that
share: 0.115 log points on expiry days before the auction and 0.178 in 2025 onward (column
`log_rv_overstatement_iid_model`); on other days the autocorrelation is not negative, so no overstatement.

**What it changes.** For F2: noise is not higher after negative-Pro evenings and there is no bounce at 14:59, so
microstructure noise does not explain F2. For the expiry-day premia (outside F2): the last 30 minutes of expiry
days do show noise, mostly from 2025 (the Tuesday regime), so the one-minute end-of-day expiry premium is
overstated by roughly that amount relative to other days. The referee's M10 request for noise-robust
estimators (pre-averaging, realized kernels or subsampled 2- to 5-minute returns) still needs to be done for
the expiry-premium tables; this check only sizes the problem.

---

## 8. Pro's raw positions instead of the GEX sign (M1(a))

This item was added after the first seven had been run and read, so it is post hoc twice over. It has its own
script, `experiments/referee_f2_positions.py` (tests: `tests/test_referee_f2_positions.py`), which reports 381
further p-values, none adjusted.

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/referee_f2_positions.py     # about 75 seconds

**What the referee asked.** Estimate F2 directly on Pro's net call, put and index-futures positions, and the
same for FII and Client, so readers see what the signal is. Does the gamma weighting add anything?

**What we did.** From the participant file (`data/raw/participant_oi`, 944 files) we took each category's net
position at each evening's close, as a share of all open contracts of that instrument: net index calls
((long - short) / all open index calls), net index puts, net index futures, calls minus puts (a bullish option
stance is long calls and short puts) and calls plus puts. We added calls plus puts ourselves: a category's GEX is
w_call x C + w_put x P, and C and P are of similar size, so the sign of GEX should be close to the sign of
w_call + w_put. The file pools all index options and futures (every index, every expiry), so these are not
Nifty-only positions (the referee's M2.1). Each position entered F2 in place of N in two forms:

* **sign**: 1 if the category is net short. F2's direction is a negative coefficient, as for N.
* **level**: the position standardized over the regression sample. The coefficient is the change in the
  r_rod slope per standard deviation more net long, so here F2's direction is a **positive** coefficient.

Inference as in Section 3: Newey-West, wild cluster bootstraps by episode (runs of evenings with the same sign
of that position) and by quarter, and the 60-day block permutation (t and coefficient). The "largest robust p"
below is the largest of the four simulated p-values. Pro was run on all three samples, FII and Client on the
full sample only. Then horse races (Pro, full sample): the GEX state and raw states in one regression, each
with its own level and r_rod interaction. Their inference is Newey-West and wild cluster bootstraps by quarter
and by joint episode (a new episode starts when any of the signs changes). For several raw states together we
also give a joint Wald test, both the Newey-West version and a wild cluster bootstrap of the clustered Wald
statistic. No permutation is given for the horse races: moving one state while holding the other fixed breaks
their dependence and does not test that null. Input checks (`item8_input_checks.csv`): the call and put shares
equal `experiments/dealer_sign.py` exactly, the Pro GEX sign equals the N of Sections 1 to 7 on all 932 days,
and every raw position is available on all 932 days.

**Result (REAL DATA, `results/referee_f2/item8_positions_describe.csv`, `item8_positions_f2.csv`,
`item8_horse_race.csv`; log `results/referee_f2_positions_stdout.txt`).**

What the Pro positions look like on the 932 F2 days (previous evening):

| Pro position | Net short: 2023 / 2024 / 2025 / 2026 | Same sign as GEX | Correlation with GEX / gross |
|---|---|---|---|
| net index calls | 87.2% / 64.9% / 40.1% / 36.6% | 78.5% | 0.81 |
| net index puts | 75.7% / 49.4% / 21.1% / 15.6% | 75.4% | 0.70 |
| net index futures | 66.4% / 65.7% / 49.8% / 29.3% | 60.1% | 0.26 |
| calls minus puts | 60.4% / 61.2% / 60.7% / 64.9% | 53.1% | 0.10 |
| calls plus puts | 94.0% / 57.6% / 23.1% / 18.1% | 98.3% | 0.994 |
| GEX / gross (the F2 state) | 94.9% / 59.2% / 23.9% / 19.0% | 100% | 1 |

The GEX sign and the sign of calls plus puts differ on 16 of the 932 evenings.

F2 with each Pro position, full sample:

| Pro position | Form | F2 coefficient | Newey-West p | wild episode p | wild quarter p | block permutation p (t / coef) | Largest robust p |
|---|---|---|---|---|---|---|---|
| GEX (the published state) | sign | -0.0680 | 0.0002 | 0.0066 | 0.014 | 0.0008 / 0.0024 | 0.014 |
| GEX (the published state) | level | +0.0389 | 0.00001 | 0.023 | 0.026 | 0.0014 / 0.0012 | 0.026 |
| calls plus puts | sign | -0.0632 | 0.0008 | 0.0098 | 0.021 | 0.0052 / 0.0082 | 0.021 |
| calls plus puts | level | +0.0398 | 0.00002 | 0.023 | 0.027 | 0.0008 / 0.0006 | 0.027 |
| net index calls | sign | -0.0502 | 0.0060 | 0.017 | 0.074 | 0.027 / 0.049 | 0.074 |
| net index calls | level | +0.0324 | 0.0003 | 0.0042 | 0.033 | 0.0026 / 0.0052 | 0.033 |
| net index puts | sign | -0.0529 | 0.025 | 0.083 | 0.12 | 0.059 / 0.032 | 0.12 |
| net index puts | level | +0.0335 | 0.0043 | 0.031 | 0.038 | 0.020 / 0.0044 | 0.038 |
| net index futures | sign | -0.0517 | 0.023 | 0.022 | 0.036 | 0.053 / 0.032 | 0.053 |
| net index futures | level | +0.0155 | 0.16 | 0.19 | 0.19 | 0.24 / 0.23 | 0.24 |
| calls minus puts | sign | -0.0474 | 0.016 | 0.031 | 0.073 | 0.042 / 0.064 | 0.073 |
| calls minus puts | level | +0.0072 | 0.43 | 0.46 | 0.43 | 0.49 / 0.58 | 0.58 |

Pro by half (F2 coefficient, largest robust p):

| Pro position | Form | 2023 to 2024 | 2025 onward |
|---|---|---|---|
| GEX | sign | -0.0644 (0.20) | -0.0739 (0.042) |
| GEX | level | +0.0309 (0.29) | +0.0434 (0.0026) |
| calls plus puts | sign | -0.0575 (0.24) | -0.0682 (0.063) |
| calls plus puts | level | +0.0303 (0.30) | +0.0430 (0.0026) |
| net index calls | sign | -0.0242 (0.65) | -0.0567 (0.076) |
| net index calls | level | +0.0316 (0.25) | +0.0314 (0.046) |
| net index puts | sign | -0.0263 (0.72) | -0.0766 (0.16) |
| net index puts | level | +0.0175 (0.57) | +0.0357 (0.032) |
| net index futures | sign | -0.0756 (0.072) | -0.0285 (0.44) |
| net index futures | level | +0.0188 (0.39) | +0.0113 (0.73) |
| calls minus puts | sign | -0.0715 (0.12) | -0.0282 (0.43) |
| calls minus puts | level | +0.0165 (0.40) | -0.0016 (0.92) |

(The GEX sign in 2025 onward has largest robust p 0.042 here and 0.038 in Section 3: the same regression with
different random draws.)

FII and Client, full sample (F2 coefficient; Newey-West p; largest robust p). For Client the coefficients have
the opposite sign, as Client's positions roughly mirror Pro's and FII's:

| Position | FII sign | FII level | Client sign | Client level |
|---|---|---|---|---|
| GEX | -0.0660; 0.0001; 0.29 | +0.0049; 0.51; 0.89 | +0.0731; 0.00003; 0.18 | -0.0127; 0.0015; 0.31 |
| calls plus puts | -0.0654; 0.0001; 0.32 | +0.0046; 0.56; 0.88 | +0.0748; 0.00004; 0.21 | -0.0130; 0.0035; 0.30 |
| net index calls | -0.0030; 0.90; 0.92 | +0.0022; 0.83; 0.87 | +0.0155; 0.50; 0.55 | -0.0128; 0.11; 0.33 |
| net index puts | -0.0660; 0.0020; 0.49 | +0.0072; 0.45; 0.79 | +0.0888; 0.00001; 0.74 | -0.0166; 0.067; 0.46 |
| net index futures | -0.0541; 0.047; 0.14 | -0.0009; 0.93; 0.95 | +0.0583; 0.022; 0.079 | -0.0026; 0.80; 0.85 |
| calls minus puts | -0.0581; 0.032; 0.094 | -0.0053; 0.58; 0.69 | +0.0546; 0.0097; 0.091 | +0.0024; 0.80; 0.86 |

FII is net short index puts on 1.9% of these days and Client on 96.7%, so those two signs mark a handful of
unusual evenings; their small Newey-West p-values do not survive the robust methods.

Horse races, Pro, full sample (coefficient; Newey-West p, wild joint-episode p, wild quarter p):

| Regression | GEX state | Raw state(s) | Raw states jointly (Wald) |
|---|---|---|---|
| GEX sign + calls sign | -0.0650; 0.0013, 0.0032, 0.0066 | calls -0.0047; 0.79, 0.80, 0.80 | |
| GEX sign + puts sign | -0.0555; 0.0043, 0.022, 0.010 | puts -0.0272; 0.28, 0.31, 0.37 | |
| GEX sign + futures sign | -0.0589; 0.0005, 0.0022, 0.038 | futures -0.0385; 0.065, 0.044, 0.026 | |
| GEX sign + calls minus puts sign | -0.0610; 0.0007, 0.0040, 0.013 | -0.0288; 0.13, 0.12, 0.20 | |
| GEX sign + calls plus puts sign | -0.0735; 0.047, 0.060, 0.073 | +0.0063; 0.86, 0.88, 0.85 | |
| GEX sign + call, put and futures signs | -0.0425; 0.065, 0.094, 0.010 | calls -0.0138, puts -0.0189, futures -0.0348 | 0.26, 0.32, 0.17 |
| GEX sign + call, put and futures levels | -0.0161; 0.62, 0.66, 0.54 | calls +0.0233, puts +0.0161, futures +0.0109 | 0.21, 0.34, 0.23 |
| GEX level + call, put and futures levels | +0.0436; 0.68, 0.80, 0.68 | calls -0.0045, puts -0.0049, futures +0.0098 | 0.81, 0.86, 0.61 |
| GEX level + calls plus puts level | +0.0358; 0.58, 0.57, 0.57 | +0.0036; 0.96, 0.96, 0.96 | |

**What it changes.**

* **Which raw position carries F2.** Pro's total net index-option position, calls plus puts. Its sign gives
  -0.0632 (largest robust p 0.021) and its level +0.0398 per standard deviation (0.027), against -0.0680
  (0.014) and +0.0389 (0.026) for the GEX state. No single leg is the whole of it: net calls and net puts each
  carry part of F2 as levels (largest robust p 0.033 and 0.038) but are weaker as signs (0.074 and 0.12).
  Pro's directional option stance (calls minus puts) does not carry F2 as a level (p 0.43 to 0.58), and
  neither does the level of its index futures (p 0.16 to 0.24). The futures sign is borderline alone
  (largest robust p 0.053) and keeps part of its coefficient next to the GEX sign (-0.0385, p 0.026 to
  0.065).
* **Whether the gamma weighting adds anything.** Not measurably. The GEX state and calls plus puts have the
  same sign on 916 of 932 evenings, and their levels correlate 0.994. When the two signs race, GEX keeps
  -0.0735 and calls plus puts +0.0063. But that split comes from the 16 evenings on which they disagree, and
  the GEX p-values are 0.047 to 0.073, so it is weak evidence. When the GEX sign races the levels of calls,
  puts and futures, it falls to -0.0161 (p 0.54 to 0.66). The raw levels are then jointly insignificant too
  (p 0.21 to 0.34): the two sets carry the same information, and these data cannot separate them. The GEX
  sign beats each single raw sign (largest robust p 0.0066 to 0.038), which only says again that no single
  leg is the whole signal.
* **Halves.** As for the GEX state, nothing is robust in 2023 to 2024 (largest robust p 0.072 or more for
  every position and form). From 2025 the levels carry F2 best: calls plus puts +0.0430 (largest robust p
  0.0026, the same as the GEX level), net puts +0.0357 (0.032) and net calls +0.0314 (0.046).
* **FII and Client.** No position of either survives the robust methods (largest robust p 0.079 or more), so
  the raw positions confirm Section 4: the pattern belongs to Pro.

In plain words: F2 is "the close reverses the day's move more after evenings on which Pro is, on balance, net
short index options". Weighting Pro's calls and puts by the front chain's gamma changes that state on 16
evenings in four years and adds nothing we can detect. This answers M1(a). It supports the referee's
renaming request (M1(c)), for example "Pro net index-option position". Because the file pools all indices
and expiries, the position is Pro's book across all index options, not its Nifty front-expiry position.

---

## What changes in the claims, point by point

| Referee point | Change |
|---|---|
| M2.5 | Lead with the unconditional test: under `short_all` hedging theory predicts momentum; the slope is -0.0294 (block bootstrap 95% interval -0.048 to -0.005) and does not rise with gamma, so the prediction fails. Describe the reversal as a full-sample and expiry-day pattern (not significant in 2025 onward). |
| M2.2, M3 (state) | Describe N as a slow Pro net-position state (95% of 2023 evenings, about 20% from 2025, two quarters carrying most 2025 to 2026 negatives); report the run and effective-sample table. |
| M3 (inference) | Replace "holds in both halves" by "robust in the full sample (largest robust p 0.012) and in 2025 onward (largest robust p 0.038); not robust in 2023 to 2024 (p 0.095 to 0.18 under episode or quarter clusters and block permutations, gone with controls)". |
| M2.4 | F2 exists only under the Pro rule: FII, FII plus Pro and Client mirror fail robust inference and vanish with controls; DII is never negative; `short_all` cannot carry an interaction. |
| M3 (alternatives) | None of the 15:15 split, the first bar, month or quarter effects, a January-to-March interaction, SEBI phase dummies or Sensex days removes F2 (full-window coefficients -0.053 to -0.079 under these controls; the split puts -0.040 before 15:15 and -0.028 after). Square-off and close-benchmarked flows cannot be excluded with these data, only shown not to be the whole effect. |
| M13 (angles) | "169 degrees apart" becomes "outcome (b) points to reverse Pro; outcome (a) is not identified, and a test that allows for that cannot reject a common direction (p 0.07 to 0.28)". |
| M10 | Last-30-minute 1-minute returns on expiry days are negatively autocorrelated (-0.054, -0.082 from 2025), so one-minute end-of-day expiry premia carry noise; F2 does not. |
| M1(a) | State F2 on Pro's raw net index-option position: the sign of net calls plus net puts gives -0.0632 (largest robust p 0.021) against -0.0680 for the GEX sign; the two agree on 916 of 932 evenings, and a horse race cannot separate them, so the gamma weighting adds nothing measurable. No single leg (calls, puts, futures, calls minus puts) is the whole signal, and no FII or Client position carries it. |

## Not done here

* M1(a) on Nifty-only positions: not possible with these data, because the participant file pools all indices
  and expiries.
* M10: noise-robust realized variance for the expiry premia (only the noise diagnostic is here).
* Verification of the SEBI and Sensex dates against primary documents (hosts blocked from this container).

## What F2 now means

F2 is a real, persistent pattern in Nifty's last 30 minutes, but it is a statement about NSE's Pro category, not
about dealer gamma. After evenings on which Pro is, on balance, net short index options (its net calls plus net
puts across all indices are negative), the close reverses the day's move by about 7 bp more per 100 bp.
Weighting those positions by the front chain's call and put gamma, as the GEX sign does, changes the state on
16 of 932 evenings and adds nothing measurable (Section 8). The
clean hedging test that needs no proxy (dealers short everything, so momentum) fails: there is no momentum, and
more gamma brings none. The negative-Pro state is a slow regime (almost always on in 2023, about one evening in
five from 2025), so the honest sample is a few dozen episodes, not 466 days. With inference that respects
that, F2 holds in the full sample and in 2025 onward but not inside 2023 to 2024. It survives the 15:15 split,
the first-bar check, seasonal and regulatory controls and Sensex days, and no other participant category
reproduces it. What drives it (Pro's own trading near the close, regime shifts in who provides liquidity, or
something correlated with both) is not identified by these data.
