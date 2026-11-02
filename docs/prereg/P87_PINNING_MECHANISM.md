# P87 pre-registration: which market-structure variable explains when expiry-day pinning appears and disappears?

**Status.** Written on 2 November 2026 (phase 1), before any outcome data were used for this question, and committed before phase 2 runs. Phase 2 files: `experiments/pinning_mechanism.py`, `results/pinning_mechanism/`, `results/pinning_mechanism_stdout.txt`, `docs/PINNING_MECHANISM.md`, and tests on SYNTHETIC data in `tests/test_pinning_mechanism.py`. All results will be REAL DATA.

**Prior knowledge.** I had read `docs/PINNING_STUDY.md` and `docs/SENSEX_CONFOUND.md`, including the regime gaps, the half-year table, the null result for Nifty monthly expiries, and an exploratory split by "OI share at the strike nearest the close" that pointed the wrong way (+1.38 points, p = 0.19). For phase 1 I looked only at column names, at units on two non-expiry days (5 and 8 July 2024), and at Bank Nifty's expiry dates after 3 August 2026. Open interest (OI) is in shares, `CONTRACTS` and `TtlTradgVol` count contracts, and `VAL_INLAKH` times 100,000 and `TtlTrfVal` both equal contracts times lot size times (strike plus premium). No candidate was computed on the panel, and no closes or distances were looked at.

## 1. Outcome and panel

**Outcome.** y is the distance from NSE's official close (`data/raw/index_daily.csv`) to the nearest strike (`pinning_study.dist_to_grid`), divided by the strike spacing g (50 for Nifty, 100 for Bank Nifty). y runs from 0 to 0.5. With no pinning it averages 0.25, with a standard deviation (SD) of 0.144, which is 7.2 Nifty points. Pinning means a smaller y on expiry days.

**Panel.** The panel is the pinning study's day panel (`pinning_study.build_day_panel`, imported rather than copied). It runs from 3 January 2023 to 27 October 2026: 1,878 index-days, of which 322 are expiry days (200 Nifty, 122 Bank Nifty). The outcome is the official close, which is available to 2 November 2026 for both indices. Minute bars (Bank Nifty's end on 22 April 2026) are used in one placebo only.

**Cells.** A cell is one index within one regime, with the closing auction split off. Expiry days per cell:

* Nifty: multi-weekly to 19 November 2024 (98); single weekly Thursday to 31 August 2025 (41); Tuesday (48); Tuesday under the auction from 3 August 2026 (13).
* Bank Nifty: weekly options to 19 November 2024 (98); monthly only (21); monthly under the auction (3).

## 2. Candidate mechanisms

**Notation.** t is the day and E is the index's next expiry (E = t on expiry days). "The contract" means the options expiring on E, taken from the last bhavcopy before t with `pinning_study.chain_before`, which also handles holiday re-dating. S is the previous official close. Each candidate is standardized: we subtract its mean and divide by its SD over the index's expiry days. All candidates are known at the previous close, except C2d.

* **C1, OI concentration (primary).** The contract's OI (calls plus puts) at the two strikes around S (g times floor(S/g), and that plus g), divided by its total OI. *Story:* when many open options sit next to where the index trades, the hedging of their holders and sellers (and any wish to steer the settlement price) is concentrated there, so the close is more easily pulled onto a strike (Avellaneda and Lipkin 2003).
* **C2, volume share.** The contract's notional turnover on the previous day, divided by that of all options on the same index. Turnover is `VAL_INLAKH` times 100,000 before July 2024 and `TtlTrfVal` after. The files hold only Nifty and Bank Nifty. C2d uses day t instead; it is a sensitivity check only, since pinning could cause that trading. *Story:* when most trading is in the contract about to settle, flows that settle at the official close dominate.
* **C3, contract size.** The log of lot size times S. The lot size is `NewBrdLotQty` (July 2024 on). Before that it is the greatest common divisor of the contract's positive OI values, checked against turnover divided by (contracts times (strike plus premium)). *Story:* big contracts keep out small retail traders, who mostly buy options. That leaves dealers less short gamma, and short-gamma hedging pushes prices away from strikes.
* **C4, weekly expiries per week.** The number of index option contracts with a weekly (not monthly) expiry in the calendar week, on NSE and BSE. Nifty and Bank Nifty come from the bhavcopies. Sensex comes from `results/sensex_confound/sensex_expiry_days_used.csv`, built from the reported calendar in `data/external/sensex_expiry_calendar.csv`. Other indices come from a new `data/external/other_weekly_expiries.csv`, with a source and a verified or reported flag on each row. *Story:* many weeklies spread traders and hedging capital thin.
* **C5, expiring OI relative to the next expiry.** The log of (the contract's OI divided by the OI of the next later expiry), both from the previous day. *Story:* positions held into settlement, not rolled, mean more hedges unwind at the close.
* **C6, strike spacing relative to the typical move.** The log of g / (S times the root mean square of the 20 daily log changes of the official close up to the previous day). January 2023 uses `data/raw/extra_2022`. *Story:* if the index moves only a strike or two a day, a modest pull can land it on a strike. If it moves through many strikes, noise swamps the pull.
* **A1, enforcement (not market structure; descriptive only).** 1 from 4 July 2025, after SEBI's interim order in the Jane Street matter (reported date 3 July 2025). *Story:* if trading aimed at the settlement price caused part of the pinning, scrutiny would end it.

Every candidate predicts **more pinning at higher values**, except C4 (fewer weeklies, more pinning). Robustness variants of C1: C1g, gross gamma at S with 30 minutes left (`pinning_study.dealer_gamma_at`, weights 1) divided by its value if all OI sat at S, and a window of two spacings.

## 3. The test

For each candidate z separately, we fit an ordinary least squares regression on all index-days:

> y = a[cell] + b[cell] times E + d times z + gamma times E times z + m[index] times M + n[index] times E times M + error

The terms are:

* E: 1 on the index's expiry day.
* a and b: each cell's own level and own expiry effect (fixed effects).
* M: 1 for a monthly contract, which expires with the index futures (`data.futures_expiry_dates`).
* **gamma:** how much more y falls on expiry days per unit of z than on ordinary days.

Because each cell has its own expiry effect, gamma uses only differences between days within a cell. Standard errors are clustered by calendar week (CR1), and p-values use a t distribution with (weeks minus 1) degrees of freedom. A day missing a candidate is dropped for that candidate only.

**Primary hypothesis.** On expiry days, a larger share of the expiring contract's OI at the two strikes around the previous close goes with an official close nearer a strike, beyond any such link on ordinary days of the same index and regime.

**Primary test.** gamma for C1 in this model, with H1 supported if gamma < 0 and the two-sided week-clustered p < 0.05. A within-cell permutation p is reported as a check that decides nothing (C1 shuffled within cell and expiry status, 10,000 draws, seed 8701).

**Secondary family.** gamma for C2, C3, C5 and C6, two-sided, with Holm's correction over the four. Holm multiplies the smallest p by 4, the next by 3, and so on, never letting an adjusted value fall below the one before. This keeps the chance of any false positive at 5%. C4 and A1 are descriptive. Exploratory only: slopes for each index, C1g, C2d, the wider window, no monthly controls, dropping auction days, holiday-shifted expiries or re-dated contracts, and clustering by month.

## 4. Identification

**Within and between regimes.** Only five cells have 20 or more expiries, and the regime pattern was found on these same data. A regime-level variable that "fits" five known gaps is a description, not a test, so the confirmatory tests use within-cell variation only. C1, C2, C5 and C6 vary day to day within cells. C3 varies within cells only through lot changes and index drift. C4, A1 and the weekday vary only between regimes.

**Public facts already limit C3 and C4.** SEBI's circular of 1 October 2024 set one weekly expiry per exchange and a Rs 15 lakh minimum contract size from 20 November 2024. Both still held after the move to Tuesday. So the weekly count (Nifty plus Sensex) and contract size are about the same in the Thursday and Tuesday regimes, yet pinning switched off (published p = 0.0008). Neither variable can explain the disappearance. At Bank Nifty's November 2024 switch both moved toward more pinning, but the published gap shrank. Phase 2 confirms these values from the files.

**Between regimes: prediction, not fitting.** From the within-cell estimates (d, gamma) and the cell means of z on expiry and other days, each cell gets a predicted gap: d times (mean z on expiry days minus mean z on other days) plus gamma times (mean z on expiry days). Predicted and observed changes are compared for three contrasts, without auction days: Nifty multi-weekly to Thursday, Nifty Thursday to Tuesday, and Bank Nifty weekly to monthly. The share explained is s = sum(w times pred times obs) / sum(w times obs squared), with w = 1 / (bootstrap variance of obs). The bootstrap resamples calendar weeks within cells, 2,000 draws, seed 8703.

**Trends and other changes.** Pinning built up within the multi-weekly regime through 2024, so a candidate trending over those months could give a false slope (placebo P5 checks this). Around 1 September 2025 the weekday changed, Sensex moved to Thursday, and the enforcement order came two months earlier. One switch cannot separate these. A1 is described from the 8 Thursday expiries after 3 July 2025 only.

## 5. Power (published numbers only)

The pinning study's half-year standard errors imply an SD of 5.7 to 7.7 Nifty points on expiry days, and its P1 standard error (0.58) matches 7.2 times sqrt(1/200 + 1/739) = 0.57, so we use 7.2 points (0.144 g). With 322 expiry days, 1,556 other days and a within-cell SD of z of 1, gamma's standard error is 0.144 times sqrt(1/322 + 1/1556) = 0.0088 g.

**The minimum detectable effect (80% power, two-sided 5%) is 2.8 times that: 0.025 g, or 1.2 Nifty points (2.5 Bank Nifty points) per within-cell SD.** That is a correlation of about 0.17, a quarter of the 4.52-point Thursday-regime effect. The first Holm step needs about 1.5 points. Both sizes double if z varies within cells only half as much as overall. Of the 322 expiries, 196 are in the two early weekly cells and 41 in the pinned Thursday cell. The published regime contrasts have standard errors of about 1.5 and 1.6 Nifty points and 3.6 Bank Nifty points, so s will be imprecise.

## 6. Placebos and falsification

Each placebo reruns the model with one change. A flag means the candidate does not act through expiry-day settlement.

| Placebo | Change | Prediction | Flag if |
| --- | --- | --- | --- |
| P1, ordinary days | Non-expiry days, "day before expiry" as E; also d | Near 0 | Same sign as gamma, p < 0.05 |
| P2, earlier price | 14:30 price (14:29 bar) instead of the close | Near 0 | Same sign, p < 0.05, at least as large as with the close |
| P2b, 15:00 price | 14:59 bar; reported only (the Nifty Thursday gap had mostly formed by 15:00) | No larger than with the close | (none) |
| P3, fake strikes | Mean distance to strikes shifted by g/4; a pull equally likely from either side leaves it exactly at g/4 | 0 | p < 0.05 |
| P4, half-strike grid | Share within g/10 of a half-way point (the mean distance there is just 0.5 g minus y) | gamma at most 0 | gamma > 0, p < 0.05 |
| P5, trend | Separate expiry effect per cell and half-year | Same sign, at least half the size | Otherwise |

P2 bars run to 2 November 2026 for Nifty and 22 April 2026 for Bank Nifty. Placebo permutations use seed 8704.

**What would falsify each mechanism?**

* C1, C2, C5 and C6 are falsified by a wrong-sign gamma with p < 0.05 (Holm for the secondaries).
* For C2, an effect only in C2d points to reverse causation.
* C3 and C4 are already ruled out for the disappearance (Section 4), unless the files show that their Thursday and Tuesday values differ.

## 7. Decision rule for "explains"

* **Explains when pinning appears and disappears:** all three hold.
  1. gamma < 0 with p < 0.05 (C1), or Holm p < 0.05 (others).
  2. No placebo flag.
  3. Correct predicted sign for both Nifty contrasts (more pinning from multi-weekly to Thursday, less from Thursday to Tuesday), and s of at least 0.5.
* **Matters day to day, not for regimes:** 1 and 2 hold, 3 fails.
* **Suspect:** 1 holds, but a placebo flags.
* **Not supported:** 1 fails.
* C4 and A1 can only be "consistent" or "inconsistent" with the regime pattern.

If nothing explains, the write-up says so: public end-of-day data cannot tie the timing of pinning to a measured market-structure variable.

## 8. Closing auction, seeds and procedure

* **Closing auction.** From 3 August 2026 the official close comes from a closing auction (NSE/CMTR/73362, NSE/CMTR/75479). Auction days are their own cells, and the check that the settlement price equals the official close is rerun. P2 uses 14:30 and 15:00, which are still in continuous trading (the minute index freezes from about 15:15), and no bar after 15:14 is used. A sensitivity check drops all days from 3 August 2026.
* **Seeds.** `numpy.random.default_rng` with 8701 (primary), 8702 (secondary), 8703 (bootstrap) and 8704 (placebos), and 10,000 permutation draws.
* **Order.** The script writes `candidates.csv` and `candidate_checks.csv` before joining any outcome. The checks cover lot-size rule agreement with `NewBrdLotQty` (at least 99%, otherwise the turnover rule is used), missing counts and within-cell SDs. It then writes `primary_test.csv`, `secondary_tests.csv`, `between_regime.csv`, `placebos.csv`, `robustness.csv`, `design_log.csv` and `fig_within_slopes.png`. Command: `OMP_NUM_THREADS=1 python experiments/pinning_mechanism.py --raw data/raw --out results/pinning_mechanism`.
* **Deviations.** Any change after phase 2 starts goes into `design_log.csv` and a "Deviations" section of `docs/PINNING_MECHANISM.md`, next to the result as specified here.
