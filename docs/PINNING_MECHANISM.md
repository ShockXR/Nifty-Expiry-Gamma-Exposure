# What decides when expiry-day pinning appears? Five market-structure candidates, tested as pre-registered (P87)

**REAL DATA.** NSE F&O bhavcopies and NSE official index closes, 3 January 2023 to 27 October 2026, plus third-party 1-minute bars (one placebo only). Every number below comes from a file written by `experiments/pinning_mechanism.py` (folder `results/pinning_mechanism/`, log `results/pinning_mechanism_stdout.txt`). The only SYNTHETIC data are in the unit tests (`tests/test_pinning_mechanism.py`).

**Pre-registration.** The plan was written before any outcome was looked at and committed to git as `docs/prereg/P87_PINNING_MECHANISM.md`, commit `e50a77506692aaf9b53cfca6c3cff6010308d3f7` (2 November 2026, 22:39 UTC). The script was written afterwards and follows it. Section 6 lists every place where the plan left a choice open, and the two corrections made between the first and the final run. There are no deviations from the analysis plan.

Written for Shreyan Thanki, to discuss with Prof. Piyush Pandey. No finance or statistics background is assumed.

## Findings first

1. **The pre-registered primary mechanism fails, and it fails in the opposite direction.** We tested whether expiry closes land nearer a strike when more of the expiring contract's open interest (OI) sits at the two strikes around the previous close. They land *farther* from a strike: +0.022 of the strike spacing per standard deviation (SD) of that share (95% interval +0.005 to +0.039). That is +1.1 Nifty points per SD (interval +0.3 to +2.0), week-clustered p = 0.0099, permutation p = 0.013. By the pre-stated rule, a significant slope of the wrong sign **falsifies** this mechanism. An exploratory check in the earlier pinning study had already pointed this way (+1.38 points, p = 0.19).
2. **One of the four secondary candidates passes, but it does not explain the timing of pinning.** When the expiring contract holds more OI relative to the next expiry (C5), the expiry close is nearer a strike: -0.028 of the spacing per SD (interval -0.047 to -0.009), which is -1.4 Nifty points, Holm-adjusted p = 0.0099. It passes every placebo. But its regime averages move the wrong way: they predict *less* pinning in the Thursday regime, where pinning was strongest, and no change at the move to Tuesday, where pinning vanished. The pre-registered verdict is **"matters day to day, but not for the regime pattern"**.
3. **Contract size (C3) has a significant slope of the wrong sign** (+0.041 per SD, Holm p = 0.0037): bigger lots go with closes farther from strikes. Between regimes it fails as well. Contract value per lot roughly doubled at the November 2024 switch (Nifty 8.7 to 16.0 lakh rupees on expiry days), just when Nifty pinning was strongest, and was about the same in the Thursday and Tuesday regimes (16.0 against 17.1 lakh), when pinning switched off.
4. **The expiring contract's share of trading (C2) and strike spacing relative to the daily move (C6) show nothing** (Holm p = 0.91 and 0.56). The same-day version of the trading share does predict pinning (exploratory, p = 0.0014), but the previous day's version does not. The plan said in advance that this pattern points to reverse causation: pinned days attract trading, rather than trading causing pinning.
5. **The weekly-expiry count (C4) behaves as the plan predicted from public rules.** It fell from about 5 weekly expiries per week to about 1.5 in November 2024 and did not change at the move to Tuesday. So it fits the appearance of Nifty pinning in November 2024, but not its disappearance in September 2025 or Bank Nifty's loss of pinning (descriptive only).
6. **Enforcement (A1, descriptive).** In the 8 Thursday expiries after SEBI's interim order of July 2025, Nifty still pinned (-3.9 points against -4.6 before), so the order does not obviously mark the end of pinning. The 8 days are far too few to be sure.

**In one sentence:** none of the five measured market-structure variables explains when NSE expiry-day pinning appears and disappears. The pre-registered favourite (OI piled up near the money) goes the wrong way, and the one variable that helps day to day (OI held into expiry rather than rolled) cannot account for the regime pattern.

## 1. The question in plain words

**Pinning** means the index closes unusually near an option strike on expiry day. Strikes are every 50 points for Nifty and every 100 for Bank Nifty. The official close is the price at which every expiring option is settled. The earlier study (`docs/PINNING_STUDY.md`) found pinning for Nifty from 20 November 2024 to 31 August 2025, when Nifty had the only NSE weekly expiry, on Thursdays (4.52 points nearer a strike than on other days). There was none after the move to Tuesday, and Bank Nifty pinned while it had weekly options. Dealer-gamma proxies did not explain it.

**P87 asks** whether a feature of how the market is organized ("market structure") explains this timing. If the market is busier or more crowded in a particular way, does pinning appear, and does it disappear when that feature goes away?

## 2. What was measured

Every candidate is known by the previous evening, except one variant. It is built from public NSE files: the bhavcopy of the last trading day before the expiry day, the previous official close, and the expiry calendars. Full definitions are in the pre-registration. Each was standardized: we subtracted its mean and divided by its SD over that index's expiry days, so "1 SD" means a typical-sized difference.

| Code | Variable | Economic story | Predicts more pinning when |
| --- | --- | --- | --- |
| C1 (primary) | Share of the expiring contract's OI at the two strikes around the previous close | Hedging, and any wish to steer the settlement price, is concentrated where options are piled up next to the index (Avellaneda and Lipkin 2003) | higher |
| C2 | Expiring contract's share of the index's option turnover, previous day | Flows that settle at the official close dominate when most trading is in the expiring contract | higher |
| C3 | Contract size: log of lot size times index level | Bigger contracts keep out retail option buyers, which leaves dealers less short gamma | higher |
| C4 | Weekly index option expiries in the calendar week, NSE and BSE | Many weeklies spread traders and hedging capital thin | lower |
| C5 | Log of expiring OI over next-expiry OI, previous day | Positions held into settlement rather than rolled mean more hedges unwind at the close | higher |
| C6 | Log of strike spacing over the typical daily move (20-day root mean square of daily changes) | A coarse grid relative to daily moves lets a small pull land the index on a strike | higher |
| A1 | After SEBI's interim order in the Jane Street matter (from 4 July 2025) | Scrutiny would end pinning caused by trading aimed at the settlement price | lower (descriptive) |

**Sources for C4.** Nifty and Bank Nifty dates come from the bhavcopies. Sensex dates come from the reported calendar of `docs/SENSEX_CONFOUND.md`. Fin Nifty, Midcap Select, Bankex and Sensex 50 come from a new file, `data/external/other_weekly_expiries.csv`. Every row of that file is "reported" (built from web search summaries of NSE and BSE notices), because BSE's servers could not be reached.

## 3. How the test works

**The outcome** is the distance from the official close to the nearest strike, as a share of the strike spacing. It runs from 0 (on a strike) to 0.5 (half way between strikes) and averages 0.25 with no pinning. Multiply by 50 for Nifty points or by 100 for Bank Nifty points.

**The regression.** All 1,878 index-days (322 expiry days) are used, and each candidate gets its own regression. Days are grouped into **cells**: one index in one regime, with the closing-auction period (from 3 August 2026) split off, which gives seven cells. Each cell gets its own average distance and its own expiry-day effect ("fixed effects"). The coefficient **gamma** measures how much more the distance falls on expiry days per SD of the candidate than it does on ordinary days.

Because every cell has its own expiry effect, gamma is estimated **only from differences between days in the same regime**. That is deliberate. Only five regimes have enough expiries, and the regime pattern was found on these same data. A variable that merely lines up with five known numbers proves little.

**Uncertainty.** Days in the same calendar week may be related, so standard errors are "clustered" by week. As a check, the candidate was also shuffled among days of the same cell and expiry status 10,000 times; the permutation p-value is the share of shuffles with a gamma as large as the real one.

**Many tests.** The four secondary candidates are corrected with Holm's method, which keeps the chance of any false positive among the four at 5%.

**From day to day to regimes.** A variable can matter day to day without explaining the regime pattern. So the plan predicted each regime's expiry gap from the within-regime slopes and the regime's average value of the candidate, and compared the predicted changes with the observed ones for three switches: Nifty multi-weekly to Thursday, Nifty Thursday to Tuesday, and Bank Nifty weekly to monthly. "Share explained" s is 1 if the predictions match and 0 if they explain nothing.

**Decision rule (pre-registered).** A candidate "explains" only if all three hold:

1. gamma is negative with p < 0.05 (Holm for the secondaries);
2. no placebo flags it;
3. its predictions get both Nifty switches right, with s of at least 0.5.

## 4. Results (all REAL DATA)

### 4.1 Data checks

* **Panel.** All seven cells have exactly the expiry counts stated in the plan (Nifty 98, 41, 48 and 13; Bank Nifty 98, 21 and 3), and there are 1,878 index-days with no missing candidate values (`candidate_checks.csv`).
* **Lot sizes.** The plan's first rule for older files, the greatest common divisor of OI, agreed with NSE's own `NewBrdLotQty` on only 89.1% of 1,138 newer contract-days, below the 99% bar. So, as the plan required, the turnover-based rule was used for all 740 older contract-days; it agreed with `NewBrdLotQty` on 100%. The lot-size history it gives matches the known revisions: Nifty 50, 25 (April 2024), 75 (from late 2024) and 65 (end of 2025); Bank Nifty 25, 15 (July 2023), 30, 35 and 30.
* **Monthly expiries.** The monthly flag (the contract expires with the index futures) agrees with the pinning study's label on 198 of 200 Nifty and 118 of 122 Bank Nifty expiry days.
* **Settlement.** The settlement price of every expiring option equals the official close on all 322 expiry days, including the 16 under the closing auction (`settlement_check.csv`).
* **Within-regime spread of C1.** In the cells with 13 or more expiries, the SD of C1 within the cell on expiry days is 0.75 to 1.06 times its overall SD, so the planned power (Section 5 of the plan: 1.2 Nifty points per within-cell SD) applies roughly as stated.

### 4.2 Primary test: C1, OI concentration near the money

| | Value |
| --- | --- |
| gamma (share of strike spacing per SD) | **+0.0224** (95% interval +0.0054 to +0.0393) |
| In Nifty / Bank Nifty points per SD | +1.12 / +2.24 |
| Week-clustered p / permutation p | **0.0099** / 0.013 |
| Same slope on ordinary days (d) | -0.0017 (p = 0.64) |
| Days, expiry days, weeks | 1,878, 322, 200 |
| **Decision** | **Hypothesis not supported; mechanism falsified (significant wrong sign)** |

Source: `primary_test.csv`, `decisions.csv`.

The size is close to the planned minimum detectable effect (1.2 points), so the test had the power the plan expected. The sign is the opposite of the hypothesis. On expiry days with more OI next to the previous close, the close lands about 1.1 Nifty points *farther* from a strike per SD. On ordinary days there is no such link. Placebo P4 agrees: with more OI near the money, more expiry closes sit near a half-way point between strikes (+0.080, p = 0.0017). That is the mirror image of pinning.

The result holds when monthly controls are dropped, auction days are dropped, holiday-shifted or re-dated expiries are dropped, or clustering is by month (p 0.007 to 0.04). It is similar in each index on its own (Nifty +0.021, p = 0.08; Bank Nifty +0.020, p = 0.14). The gamma-weighted and wider-window versions point the same way (p = 0.07 and 0.05). These checks are exploratory (`robustness.csv`).

### 4.3 Secondary family (Holm over four)

| Candidate | gamma per SD (95% interval) | Nifty points | p | Holm p | Permutation p | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| C2 volume share (previous day) | -0.0012 (-0.0206 to +0.0183) | -0.06 | 0.91 | 0.91 | 0.90 | not supported |
| C3 contract size | +0.0410 (+0.0170 to +0.0650) | +2.05 | 0.0009 | **0.0037** | 0.0032 | not supported; significant wrong sign |
| C5 expiring vs next-expiry OI | -0.0280 (-0.0465 to -0.0094) | -1.40 | 0.0033 | **0.0099** | 0.0031 | **matters day to day, not for the regime pattern** |
| C6 spacing vs daily move | +0.0114 (-0.0093 to +0.0321) | +0.57 | 0.28 | 0.56 | 0.22 | not supported |

Source: `secondary_tests.csv`, `decisions.csv`. The plan listed a wrong sign as a falsifier for C1, C2, C5 and C6 but not for C3, so C3 is reported as "not supported" with its wrong-sign slope stated.

**Only C5 survives Holm in the predicted direction.** It is also stable in the exploratory checks: Nifty alone -0.025 (p = 0.03), Bank Nifty alone -0.033 (p = 0.07), and p of 0.008 or below without monthly controls, without auction days, without shifted or re-dated expiries, or with month clustering.

**C3's within-regime variation comes almost entirely from a handful of lot changes and the slow drift of the index level.** That kind of variation is easily confused with other things that changed at the same time, so its wrong-sign slope should be read with care.

### 4.4 Placebos

Each placebo reruns the model with one change (`placebos.csv`; seed 8704 for the permutation checks).

| Placebo | C1 | C2 | C3 | C5 | C6 |
| --- | --- | --- | --- | --- | --- |
| P1a fake expiry (day before expiry) | +0.012 (p 0.17) | +0.009 (0.19) | +0.016 (0.30) | +0.010 (0.16) | +0.004 (0.66) |
| P1b slope on ordinary days | -0.002 (0.64) | -0.000 (0.92) | +0.002 (0.69) | +0.001 (0.55) | +0.001 (0.71) |
| P2 price at 14:30 | +0.002 (0.87) | -0.010 (0.30) | +0.001 (0.97) | -0.013 (0.14) | +0.007 (0.51) |
| P2b price at 15:00 (reported only) | -0.005 (0.64) | +0.007 (0.55) | +0.001 (0.94) | +0.010 (0.30) | +0.003 (0.75) |
| P3 fake strikes shifted by a quarter spacing | +0.017 (0.064) | +0.008 (0.38) | +0.017 (0.13) | +0.008 (0.37) | +0.003 (0.71) |
| P4 closes near half-way points | **+0.080 (0.0017), flag** | +0.010 (0.69) | +0.046 (0.12) | -0.051 (0.071) | +0.011 (0.69) |
| P5 separate expiry effect per half-year | +0.019 (0.081) | -0.008 (0.43) | +0.035 (0.13) | -0.025 (0.0043) | +0.008 (0.54) |

* **C5 passes all of them.** Its effect is absent at 14:30 and 15:00, appears in the official close, is not seen on the day before expiry, and survives half-year expiry effects (P5). The share of closes near half-way points falls (P4, -0.051, p = 0.07), as strike pinning would require.
* **C1's wrong-sign effect also exists only in the official close.** It is absent at 14:30 and 15:00.

### 4.5 Between regimes: can the slopes explain the regime pattern?

The observed change in the expiry gap is shown first, then each candidate's predicted change, in shares of the strike spacing (Nifty points in brackets for the Nifty rows). Negative means more pinning. Intervals come from a week-block bootstrap with 2,000 draws and seed 8703 (`between_regime.csv`).

| Switch | Observed | C1 | C2 | C3 | C5 | C6 |
| --- | --- | --- | --- | --- | --- | --- |
| Nifty multi-weekly to Thursday | -0.069 (-3.4) | -0.014 | -0.001 | +0.057 | +0.016 | -0.010 |
| Nifty Thursday to Tuesday | +0.115 (+5.8) | +0.002 | -0.001 | +0.013 | -0.000 | +0.002 |
| Bank Nifty weekly to monthly | +0.029 | -0.016 | -0.001 | +0.080 | +0.042 | -0.005 |
| Share explained s (95% interval) | | 0.06 (-0.04 to 0.20) | -0.00 | -0.10 (-0.47 to 0.27) | -0.03 (-0.21 to 0.16) | 0.05 (-0.04 to 0.19) |

No candidate comes close to s = 0.5. **C5 predicts the wrong direction for both Nifty switches.** The expiring contract held relatively *less* of the OI in the Thursday regime than in the multi-weekly regime (mean standardized value -0.27 against +0.32), so C5 predicts slightly *less* pinning there, not more. It predicts nothing at the move to Tuesday. **C1's predictions point the right way only because its slope has the wrong sign,** and they are tiny.

### 4.6 Descriptive: weekly count (C4), contract size by regime, enforcement (A1)

Mean values on expiry days (`candidate_checks.csv`, `descriptive_c4_a1.csv`):

| Cell | Weekly expiries in the week | Contract value per lot (Rs lakh) |
| --- | --- | --- |
| Nifty multi-weekly | 5.12 | 8.71 |
| Nifty single weekly Thursday | 1.59 | 16.00 |
| Nifty Tuesday (before the auction) | 1.54 | 17.06 |
| Nifty Tuesday (closing auction) | 1.54 | 15.31 |
| Bank Nifty weekly options | 5.12 | 8.21 |
| Bank Nifty monthly only | 0.24 | 16.28 |

As the plan stated from SEBI's 1 October 2024 rules, the weekly count and contract size barely changed between the Thursday and Tuesday regimes, so neither can explain why pinning vanished in September 2025. Under a simple consistency rule, written after the plan and labelled post hoc in `design_log.csv`, C4 fits the Nifty switch of November 2024 but neither the Thursday to Tuesday switch nor the Bank Nifty switch. Its within-regime slope is -0.027 per SD (p = 0.17), which is descriptive and not counted.

**A1.** In the Nifty Thursday regime, the expiry gap was -4.6 points before 4 July 2025 (33 expiries) and -3.9 points from 4 July 2025 (8 expiries), with no test, as planned.

### 4.7 Other exploratory results

* C2d, the expiring contract's share of trading on the expiry day itself: -0.031 per SD (-0.049 to -0.012), p = 0.0014. The previous day's share (C2) shows nothing, so by the plan's own rule this points to reverse causation: trading follows pinning.
* C1g (gamma-weighted, 30 minutes left): +0.016 (p = 0.07). C1 with a two-spacing window: +0.018 (p = 0.05). Both point the same way as C1.

## 5. What this means

* **Pinning is not "more pinning where more options sit".** The textbook mechanism, in which long-gamma hedgers pull the price toward heavily held strikes, predicts the opposite of what we find within regimes. One reading fits the data, though this study cannot test it: near-the-money options on NSE are largely sold by professionals to retail buyers. If the sellers are short gamma and hedge, more OI near the money means a stronger push *away* from strikes. That would agree with the anti-pinning seen in the Tuesday regime, but the dealer sign is not observed on NSE (see `docs/PINNING_STUDY.md`, Section 6), so this is a hypothesis, not a finding.
* **Holding positions into expiry (C5) matters a little, day to day.** It is about 1.4 Nifty points per SD, which is about a third of the Thursday-regime effect. It is not why whole regimes differ.
* **The regime pattern remains unexplained by measured market structure.** What changed on 1 September 2025 includes the weekday, Sensex's move to Thursday, and other rule changes in 2025. With one switch these cannot be separated. The two regime-level structural variables we could measure (weekly count, contract size) did not change at that date.

## 6. Deviations from the pre-registration

**Analysis plan: none.** The model, samples, candidates, standardization, clustering, seeds (8701 to 8704), numbers of draws, placebos and decision rule are as written in commit `e50a775`.

**Choices the plan left open** (all in `design_log.csv`):

1. For older lot sizes, which rule wins when the divisor and turnover rules disagree. This is moot, because the divisor rule failed the plan's 99% bar and the turnover rule was used throughout.
2. How the C4 calendar was built for indices other than Nifty, Bank Nifty and Sensex.
3. Using the holiday re-dated listing of the contract on the previous day.
4. How C2d finds the contract on the day itself.
5. P2 compares the 14:30 slope with the main slope on the full sample.
6. The definition of holiday-shifted expiries for one exploratory check.
7. A post hoc consistency rule for C4, labelled as such.

**Process notes.**

* Stage A (candidates only) was run once on its own before any outcome was joined, to check the candidates. That is how the lot-size rule switch was seen.
* The first full run labelled C3 "falsified" and lacked the pre-registered settlement-price check. Before the final run, the label was corrected to the plan's wording and the check was added; no estimate changed.
* Added output files (`decisions.csv`, `descriptive_c4_a1.csv`, `settlement_check.csv`, `run_info.json`) do not change any decision.

## 7. Limitations

* **Within-regime variation is what we can test, and it is not the same as the regime shifts.** A variable could matter between regimes through channels that do not vary from day to day. The design cannot see those, and with five regimes nothing else can.
* **C3's within-regime variation is thin** (a few lot changes and index drift), and C4 rests on a reported, not verified, calendar of other exchanges' weeklies.
* **The bhavcopies hold only Nifty and Bank Nifty,** so C2 is a share within the index, not of all index options.
* **Minute bars are third-party,** and Bank Nifty's end on 22 April 2026. They are used only in placebo P2.
* **The dealer sign is unobserved,** so the interpretation in Section 5 is untested.
* **Many exploratory numbers are reported.** Only the primary test and the Holm-corrected family are confirmatory.

## 8. Files and how to rerun

```bash
OMP_NUM_THREADS=1 python experiments/pinning_mechanism.py --raw data/raw --out results/pinning_mechanism
OMP_NUM_THREADS=1 python -m pytest -q tests/test_pinning_mechanism.py
```

The run is deterministic and takes about 3 minutes. `--stage-a-only` builds and checks the candidates without touching any outcome. Outputs are in `results/pinning_mechanism/`:

* `candidates.csv`, `candidate_checks.csv` (stage A);
* `settlement_check.csv`, `primary_test.csv`, `secondary_tests.csv`, `placebos.csv`, `between_regime.csv`, `decisions.csv`;
* `robustness.csv` (exploratory), `descriptive_c4_a1.csv`, `design_log.csv`;
* `fig_within_slopes.png`, and `run_info.json` (seeds, versions, SHA-256 of the inputs).

The log is `results/pinning_mechanism_stdout.txt`. New input: `data/external/other_weekly_expiries.csv` (reported sources, with addresses).

**References.** Avellaneda, M. and Lipkin, M. D. (2003), A market-induced mechanism for stock pinning, *Quantitative Finance* 3(6), 417 to 425. SEBI circular SEBI/HO/MRD/TPD-1/P/CIR/2024/132 of 1 October 2024. Other sources are as in `docs/PINNING_STUDY.md` and `docs/SENSEX_CONFOUND.md`.
