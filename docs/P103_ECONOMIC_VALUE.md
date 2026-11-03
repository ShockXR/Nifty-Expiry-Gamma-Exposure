# Is the hybrid's forecasting gain worth anything in money? (P103, REAL DATA)

**Everything in this document is REAL DATA**: the out-of-sample forecasts of the v2 forecasting run on 792 Nifty
test days (10 August 2023 to 2 November 2026) and NSE Nifty 50 1-minute index bars. Nothing is simulated except the
unit tests. **This is a research evaluation of the economic size of a statistical finding, not trading advice.**
The plan, the code and this write-up were produced by an AI assistant (Claude) working on the project.

* Registered analysis plan: `docs/prereg/P103_ECONOMIC_VALUE.md`, committed as
  **77b435aec493b7e0acf1eab90e51490bb9377f17** (3 November 2026, 03:38 UTC) and pushed before any outcome was
  computed. Every departure is listed under "Deviations" below.
* Script: `experiments/p103_economic_value.py`. Unit tests (synthetic data only):
  `tests/test_p103_economic_value.py` (19 tests).
* Outputs: `results/p103_economic_value/` (tables, `run_info.json` with the plan's commit, seeds and input file
  hashes, and the printed log `stdout.txt`).

```
PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p103_economic_value.py      # about 5 seconds
PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q tests/test_p103_economic_value.py
```

## The answer in short

**Registered overall verdict: "detectable but not economically meaningful"**, for both of the plan's reasons: the
fee is **tiny**, and the trading rule shows **no implementation gain** under the registered rule. **The fee over
the paper's mcsGARCH is about 0.04% of wealth a year, Rs 4,133 a year per Rs 1 crore** (size label "tiny", fixed
in advance as anything below 0.10%). Against the information-matched benchmark `mcsgarch_iv` the gain is not
significant after the Holm correction (S3 and S4, Holm p 0.096), so, as the plan requires us to say here: **the
economic value over mcsGARCH is not shown to come from the hybrid's method rather than from the implied-variance
information it is given.**

1. **The gain is real but tiny.** An investor who sizes a Nifty position by forecast volatility every 30 minutes
   is better off with the hybrid than with mcsGARCH (H1: one-sided p 8.1e-7, better on 69% of days). But the
   yearly fee the investor would pay for the switch is 0.041% of wealth, which closes only 8.7% of the gap between
   mcsGARCH and a perfect forecast. Even under the most generous values tried (annual Sharpe ratio 0.8, risk
   aversion 1) it is 0.33% a year. The 15-minute and last-hour gains over mcsGARCH are of the same size (0.040%
   and 0.037%, both significant after Holm).
2. **Most of it is information, and most of it is the open.** Against `mcsgarch_iv` (mcsGARCH whose daily level is
   the previous evening's straddle-implied variance) the fee is 0.029% (15 minutes) and 0.031% (30 minutes), with
   p 0.048 and 0.075 before and 0.096 after Holm. Without the 09:15 window it is 0.003% a year (p 0.20). Split by
   expiry regime it is -0.018% before November 2024 and +0.074% and +0.055% in the two weekly regimes since.
3. **The trading rule loses money whatever forecast it uses.** In this period Nifty fell on average between 09:15
   and 15:15 (-3.8 bp a day, most of it in the very first minute) and rose overnight (+5.1 bp a day), so a rule
   that is long only during the day lost before costs (annualised Sharpe ratio -1.18 with the hybrid). Resizing
   every 30 minutes then costs about half the capital a year (51.7%); the securities transaction tax and the
   assumed slippage are its two largest parts. Net
   Sharpe ratio -3.52, Deflated Sharpe Ratio 6e-15. It is not a strategy.
4. **Inside the rule, the hybrid does not keep risk closer to target, but it trades less.** Its daily tracking of
   the 20% volatility target is no better than mcsGARCH's (T1: p 0.81; mcsGARCH tracks slightly better). Its
   forecasts change less from one window to the next, so it trades 7.8 instead of 8.4 units of capital a day,
   which saves about 4.2% of wealth a year in costs (T2: p 3e-72). That saving is about mcsGARCH being jumpy,
   not about the hybrid being accurate: smoother forecasts (linear models, VIX) trade less still, the hybrid ranks
   27th of 33 sizing rules after costs, and holding a constant position beats them all.

## What this means for the project

* **A statistically better volatility forecast is not automatically worth money.** The headline QLIKE gains
  (5.8%, 7.6% and 6.4%) translate, under the standard yardstick of Bollerslev, Hood, Huss and Pedersen (2018), into
  fees of a few hundredths of a percent of wealth a year. The pitch and the paper should present the hybrid as a
  measurement result (better intraday volatility forecasts, mostly at the open, mostly from the option chain's
  information) and should not suggest an economic or trading value. If an economic sentence is wanted, this one is
  supported: "for a volatility-timing investor the gain is worth about 0.04% of wealth a year (realized utility,
  SR 0.4, risk aversion 2), and it is not significant against an mcsGARCH given the implied variance".
* **The economics strengthen the existing caveat about information.** The 30-minute QLIKE edge over `mcsgarch_iv`
  is significant in the headline (p below 0.001, all 24 overlapping origins); priced as utility on 12
  non-overlapping windows a day it is not (p 0.075), and it vanishes outside 09:15.
* **A finding the plan did not anticipate.** mcsGARCH's 30-minute forecasts jump around from window to window more
  than any other model except the naive ones and its own 63-day-refit variant, and a rule that follows them pays
  for it. This is a property of the
  benchmark worth one sentence; it is not evidence for the hybrid.
* **On the plan itself.** The registered decision table gave the case "T1 fails, T2 passes" no label of its own
  and calls it "no implementation gain". We keep that label as registered. Read with the other results, the
  accurate summary is: no better risk control, lower trading costs, for a reason unrelated to accuracy.

## Words used in this document

* **Realized variance (RV).** How much Nifty moved in a window: the sum of the squared 1-minute log returns.
* **Variance forecast (F).** A model's forecast of RV, made at the start of the window.
* **Volatility targeting.** Holding a position of "target volatility divided by forecast volatility": larger when
  the market is expected to be calm, smaller when it is expected to be wild, so that the risk taken stays steady.
* **Sharpe ratio (SR).** Average return divided by its standard deviation, per year. 0.4 is about what a broad
  stock index has earned historically.
* **Risk aversion (gamma).** How much the investor dislikes variance. With SR = 0.4 and gamma = 2 the best
  volatility target is SR/gamma = 20% a year.
* **Realized utility.** The investor's satisfaction (expected gain minus a penalty for risk) when they size their
  position with a forecast, measured after the fact with the realized variance. It is highest, 4% of wealth a
  year at SR 0.4 and gamma 2, when the forecast is perfect.
* **Fee.** The difference in realized utility between two forecasts: the share of wealth per year the investor
  would give up to use the better one. 0.01% of wealth is Rs 1,000 on Rs 1 crore (Rs 10 million).
* **bp (basis point).** One hundredth of a percent.
* **Diebold-Mariano (DM) test.** Is one forecast's average loss lower than another's by more than chance? Here
  the loss is the utility shortfall, averaged within each day, with a Newey-West correction for correlation
  between nearby days. One-sided: it asks only whether the hybrid is better.
* **Holm correction.** Raises p-values when several hypotheses are tested together, so that the chance of any
  false alarm stays at 5%.
* **Turnover.** Units of capital bought plus sold in a day. Every unit traded pays taxes, fees and slippage.
* **Tracking loss.** How far the strategy's realized volatility on a day is from its target, measured as the
  squared log ratio; RMSLD is the square root of its average (0.14 means a typical miss of about 14%).
* **Deflated Sharpe Ratio (DSR).** The probability that a strategy's true Sharpe ratio is above the one the luckiest
  of N worthless strategies would show. Above 0.95 counts as a pass.

## What was done (summary of the plan)

**Part A, realized utility.** For a window with forecast F and realized variance RV, the investor's realized
utility is (SR^2/gamma) x [sqrt(RV/F) - RV/(2F)] (Section 3.1 of the plan derives it from BHHP's assumptions: a
mean-variance investor and a constant Sharpe ratio; the paper's own text could not be fetched, so the exact
typography of its formula was not checked). In percent of wealth a year this is 100 x (SR^2/gamma) x the average
bracket, whatever the window length. SR = 0.4 and gamma = 2 as in BHHP; the p-values do not depend on them.
Windows: 30 minutes from 09:15, 09:45, ..., 14:45 (12 a day, no overlap; H1, S4); 15 minutes at every origin
(25 a day, no overlap; S1, S3); the last hour from 14:30 (S2). Test: the repository's one-sided DM test on
day-level averages of the utility shortfall.

**Part B, a trading rule.** Long Nifty over the same 12 thirty-minute windows, position min(4, target / forecast
volatility) with a 20% target, flat from 15:15 and overnight. Index returns stand in for futures returns. Costs
are those of P90 (securities transaction tax 1.25, 2 and 5 bp per round trip in the three periods, exchange and
SEBI charges, stamp duty, brokerage, GST, and 1 bp of slippage per side), charged as (turnover / 2) x the round-trip
cost each day, which is exact because the rule starts and ends every day flat. Four registered strategies: sized
by `hybrid_base`, by `mcsgarch`, by `mcsgarch_iv`, and a constant position of 2.

**Checks that would have stopped the run, all passed** (`checks.csv`): every day has its full set of forecast
origins; all 32 models' forecasts are positive; the hybrid's QLIKE margins over mcsGARCH on all origins match the
published -5.80%, -7.65% and -6.36% exactly; and the realized variance rebuilt from the minute bars matches the
forecast files in all 9,504 trading windows (largest relative difference 8.4e-13).

## Results, Part A: realized utility (REAL DATA)

**Table 1. Registered hypotheses, all 792 test days.** Utility and fee in percent of wealth a year at SR 0.4,
gamma 2 (perfect forecast: 4%). "Gap closed" is the fee divided by the benchmark's shortfall from 4%. Source:
`results/p103_economic_value/utility_tests.csv`.

| Test | Horizon, benchmark | Utility, hybrid | Utility, benchmark | Fee | Rs a year per Rs 1 crore | Gap closed | DM | One-sided p | Holm p | Days hybrid better |
|---|---|---|---|---|---|---|---|---|---|---|
| **H1 (primary)** | 30 min, mcsgarch | 3.5645 | 3.5232 | **0.0413** | **4,133** | 8.7% | 4.83 | **8.1e-7** | | 68.8% |
| S1 | 15 min, mcsgarch | 3.4066 | 3.3662 | 0.0404 | 4,042 | 6.4% | 4.60 | 2.5e-6 | 1.0e-5 | 69.2% |
| S2 | last hour, mcsgarch | 3.4745 | 3.4371 | 0.0374 | 3,737 | 6.6% | 3.41 | 0.00035 | 0.0010 | 53.5% |
| S3 | 15 min, mcsgarch_iv | 3.4066 | 3.3772 | 0.0294 | 2,940 | 4.7% | 1.67 | 0.048 | **0.096** | 59.2% |
| S4 | 30 min, mcsgarch_iv | 3.5645 | 3.5336 | 0.0309 | 3,090 | 6.6% | 1.44 | 0.075 | **0.096** | 60.1% |

**Table 2. Robustness (registered, no decision).** Fee in percent of wealth a year (one-sided p).

| Variant | vs mcsgarch | vs mcsgarch_iv |
|---|---|---|
| R1: windows from 09:30 (09:30, 10:00, ..., 15:00) | 0.0407 (9.7e-7) | 0.0204 (0.0059) |
| R2: all 24 overlapping origins, day averages | 0.0410 (1.6e-10) | 0.0256 (0.014) |
| R3: primary windows without 09:15 | 0.0187 (4.1e-7) | 0.0034 (0.20) |
| R4: expiry days (169) | 0.0477 (0.013) | 0.0647 (0.17) |
| R4: other days (623) | 0.0396 (6.8e-6) | 0.0217 (0.13) |
| R4: before the closing auction (729) | 0.0408 (5.6e-6) | 0.0302 (0.097) |
| R4: development, to 30 Jun 2025 (462) | 0.0467 (7.9e-5) | 0.0128 (0.32) |
| R4: hold-out, from 1 Jul 2025 (330) | 0.0338 (0.0011) | 0.0562 (0.055) |
| R4: several weekly expiries, to Nov 2024 (312) | 0.0429 (0.0067) | -0.0182 (0.78) |
| R4: one Thursday weekly (192) | 0.0457 (4.5e-6) | 0.0741 (0.071) |
| R4: one Tuesday weekly (288) | 0.0368 (0.0016) | 0.0553 (0.084) |

**Table 3. How large could the fee be?** Fee of the hybrid over mcsGARCH at 30 minutes (H1), percent of wealth a
year, for other Sharpe ratios and risk aversions (the fee is proportional to SR^2/gamma; the p-value is the same in
every cell). Source: `fee_grid.csv`.

| Annual Sharpe ratio | gamma 1 | gamma 2 | gamma 5 | gamma 10 |
|---|---|---|---|---|
| 0.2 | 0.021 | 0.010 | 0.004 | 0.002 |
| **0.4** | 0.083 | **0.041** | 0.017 | 0.008 |
| 0.6 | 0.186 | 0.093 | 0.037 | 0.019 |
| 0.8 | 0.331 | 0.165 | 0.066 | 0.033 |

Only three cells (SR 0.6 with gamma 1; SR 0.8 with gamma 1 or 2) reach the "small" band (0.10% to 0.50%); none is
"material".

**The two yardsticks agree.** On the same windows the hybrid's QLIKE is 8.06% (H1), 5.80% (S1), 6.36% (S2), 4.80%
(S3) and 6.91% (S4) lower than the benchmark's, and its utility shortfall is 8.67%, 6.38%, 6.64%, 4.72% and 6.63%
smaller (`qlike_same_windows.csv`). The ranking of all 32 models by utility follows QLIKE closely (`utility_models.csv`);
at 30 minutes the best utility belongs to `hybridiv_base`, 0.002 points above `hybrid_base`.

## Results, Part B: the trading rule (REAL DATA)

**Table 4. Base-case costs (1 bp slippage per side), all 792 days.** Profit and loss in bp of capital per day.
Source: `trading_metrics.csv`, `bootstrap_sharpe.csv`, `dsr.csv`.

| | hybrid_base | mcsgarch | mcsgarch_iv | constant (2) |
|---|---|---|---|---|
| Gross profit, bp a day | -10.26 | -11.03 | -10.14 | -7.14 |
| Costs, bp a day | 20.52 | 22.18 | 21.02 | 10.55 |
| Net profit, bp a day | -30.78 | -33.21 | -31.16 | -17.68 |
| Turnover, units a day (of which rebalancing) | 7.81 (4.08) | 8.43 (4.67) | 8.00 (4.27) | 4.00 (0) |
| Costs, percent of capital a year | 51.7 | 55.9 | 53.0 | 26.6 |
| Average position (share of windows at the cap of 4) | 2.65 (10.0%) | 2.66 (10.8%) | 2.62 (11.3%) | 2 (0%) |
| Realized volatility, percent a year (target 20) | 20.1 | 20.3 | 20.0 | 19.8 |
| Tracking loss RMSLD (days within 0.8 to 1.25 of target) | 0.141 (90.8%) | 0.139 (91.2%) | 0.144 (90.9%) | 0.386 (41.4%) |
| Sharpe ratio, gross | -1.18 | -1.26 | -1.17 | -0.85 |
| **Sharpe ratio, net** (95% bootstrap interval) | **-3.52** (-4.53 to -2.54) | -3.77 (-4.78 to -2.79) | -3.59 (-4.61 to -2.58) | -2.11 (-3.05 to -1.20) |
| HAC t of the mean net return | -6.79 | -7.29 | -6.88 | -4.33 |
| Net realized utility, percent of wealth a year (no costs) | -48.2 (3.56) | -52.4 (3.52) | -49.4 (3.53) | -23.8 (2.77) |
| Probabilistic Sharpe Ratio PSR(0) | 1.7e-10 | 7.9e-12 | 7.5e-11 | 5.8e-5 |
| **DSR (N = 4)** | **6.1e-15** | 1.5e-16 | 2.3e-15 | 5.2e-8 |

Realized volatility is computed from the realized variance; from the window returns it is 20.7%, 20.8%, 20.5% and
20.0%. Profits are added up without compounding, so the largest fall from an earlier peak, 248% of capital for
the hybrid rule (782 days, almost the whole sample), just says that it lost steadily. With costs set to zero the
net realized utilities are those in brackets, close to Part A's 3.56% (the cap changes them only slightly).

**Table 5. Cost stress grid: annualised Sharpe ratio, net.**

| | Gross | Slippage 0 | 0.5 bp | 1 bp (base) | 2 bp | 3 bp |
|---|---|---|---|---|---|---|
| hybrid_base | -1.18 | -2.63 | -3.07 | -3.52 | -4.40 | -5.29 |
| mcsgarch | -1.26 | -2.82 | -3.29 | -3.77 | -4.72 | -5.68 |
| mcsgarch_iv | -1.17 | -2.67 | -3.13 | -3.59 | -4.51 | -5.42 |
| constant | -0.85 | -1.64 | -1.87 | -2.11 | -2.59 | -3.07 |

**Table 6. Registered trading tests** (one-sided DM on daily series). Source: `trading_tests.csv`.

| Test | Comparison | Estimate | DM | p | Days hybrid better |
|---|---|---|---|---|---|
| **T1** tracking | hybrid_base vs mcsgarch | RMSLD +0.0018 (hybrid worse) | -0.87 | **0.81** | 47.2% |
| **T2** net utility, base costs | hybrid_base vs mcsgarch | +4.20% of wealth a year | 19.94 | **2.8e-72** | 89.8% |
| T1, robustness | hybrid_base vs mcsgarch_iv | RMSLD -0.0035 | 0.79 | 0.22 | 49.2% |
| T2, robustness | hybrid_base vs mcsgarch_iv | +1.27% a year | 3.68 | 0.00012 | 63.5% |
| T2, rebalancing trades only | hybrid_base vs mcsgarch | +4.02% a year | 16.30 | 5.4e-52 | 87.5% |
| T2, no costs | hybrid_base vs mcsgarch | +0.041% a year | 4.81 | 8.9e-7 | 70.3% |
| T2, slippage 0, 0.5, 2, 3 bp | hybrid_base vs mcsgarch | +2.63, +3.41, +5.78, +7.36% a year | 17.2 to 21.4 | below 1e-56 | about 90% |

Of T2's 4.20 points, 0.04 come from better forecasts (the no-cost row) and 4.16 from lower costs. The paired
bootstrap of the net Sharpe ratios gives the hybrid rule +0.25 over the mcsGARCH rule (95% interval 0.18 to 0.32).

**Table 7. Deflated Sharpe Ratio, hybrid rule.** The Sharpe ratio the luckiest of N worthless rules would show,
annualised, and the DSR. Source: `dsr.csv`.

| N | Variance of trial Sharpe ratios V | Hurdle (annualised) | DSR |
|---|---|---|---|
| **4 (registered)** | 0.0023 (the four strategies) | 0.80 | **6.1e-15** |
| 33 | 0.00062 (the 33 sizing rules) | 0.83 | 3.9e-15 |
| 4 | 1/T (pure noise) | 0.59 | 1.1e-13 |
| 33 | 1/T (pure noise) | 1.19 | 2.1e-17 |

As the power note in the plan said in advance, this verdict was close to certain: the hurdle is below 1.2 and the
rule's costs alone take more than 2 off its Sharpe ratio. It measures the cost of trading intraday, not the value
of the forecast.

### Where the losses come from (descriptive, added after the results were seen)

**Table 8. Nifty's average log return by part of the day, 792 test days** (`index_drift.csv`; plain t-statistics,
not tests of any hypothesis).

| Part of the day | Mean, bp a day | t |
|---|---|---|
| Overnight: previous day's last bar close to the 09:15 bar's open | +5.12 | 2.64 |
| 09:15 to 15:15 (the windows the rule trades) | -3.77 | -1.59 |
| of which the first minute (the 09:15 bar, open to close) | -2.92 | -4.01 |
| 15:15 to the last bar's close (not traded) | +1.17 | 2.96 |

In this period Nifty's gains came overnight and at the very end of the day, and the first minute after the open
gave back about 3 bp on average. The hybrid rule's 09:15 window alone lost 4.24 bp a day before costs, 41% of its
gross loss (`window_contributions.csv`). Part of this first-minute effect may belong to the index rather than to
anything tradable: the 09:15 opening value comes from the pre-open call auction, and a futures trader may not have
been able to buy at it.

**Turnover follows smoothness, and costs follow turnover** (`sizing_rules_33.csv`). Sizing the rule with each of
the 32 forecasting models in turn, the average absolute change of the log forecast from one window to the next is
0.39 for mcsGARCH, 0.36 for `mcsgarch_iv`, 0.34 for the hybrid, 0.28 to 0.29 for the linear models and 0.23 for VIX
(which only follows the time-of-day pattern). Net utility after base-case costs ranks the rules almost exactly by
turnover: the constant position is best (-23.8% a year), then VIX (-35.5%), the straddle-implied variance
(-35.8%) and the linear models (-44.9% to -45.3%); the hybrid is 27th of 33 (-48.2%) and mcsGARCH 31st (-52.4%).
Yet mcsGARCH, among the jumpiest, has the best daily tracking of all 33 rules (RMSLD 0.139).

## Registered verdicts

| Item (plan Section 5) | Statistic | Result |
|---|---|---|
| H1: 30 minutes, hybrid_base vs mcsgarch | p 8.1e-7 | supported |
| Fee at SR 0.4, gamma 2 | 0.041% of wealth a year (Rs 4,133 per Rs 1 crore) | **tiny** |
| S1 (15 minutes, mcsgarch) | Holm p 1.0e-5 | supported |
| S2 (last hour, mcsgarch) | Holm p 0.0010 | supported |
| S3 (15 minutes, mcsgarch_iv) | Holm p 0.096 | not supported |
| S4 (30 minutes, mcsgarch_iv) | Holm p 0.096 | not supported: value over mcsGARCH not shown to come from the method |
| T1 tracking | p 0.81 | fail |
| T2 net utility | p 2.8e-72 | pass |
| Part B | | **no implementation gain** (T1 fails) |
| Standalone DSR of the hybrid rule | 6.1e-15 | not a standalone strategy |
| **Overall** | | **detectable but not economically meaningful** (fee tiny; Part B no implementation gain) |

Source: `verdict.csv`. The plan's expectations, written before the run, were: H1 passes, the fee is between about
0.03% and 0.07% ("tiny"), turnover about 4 to 8 units a day, and the standalone DSR fails. All four held (0.041%,
7.8 units). The plan did not expect the rule to lose before costs, nor the hybrid's advantage in turnover.

## Deviations from the plan

**None in the windows, parameters, costs, tests, seeds, trial count or decision rules.** The run was made once,
after the plan's commit was pushed; the descriptive additions below were made afterwards and change no registered
number.

Points where the plan was silent, and what was done:

1. **Benchmarks for R1 to R3.** The plan named the window sets but not the benchmark; both mcsgarch and
   mcsgarch_iv are reported.
2. **Windows for R5** (every model's utility): 15 minutes at every origin, the 12 primary 30-minute windows, and
   the last hour.
3. **Days with zero realized strategy volatility** would have been dropped from the tracking test: there were none
   (all 792 days are in T1).
4. **The decision table's missing case.** T1 failing while T2 passes has no label of its own; the registered rule
   maps it to "no implementation gain", and that label is kept (see "What this means for the project").
5. **The formula's source.** As the plan warned, the paper's text could not be read (the network block persisted);
   the formula is our derivation from the assumptions confirmed through search-engine summaries, and BHHP's
   equation number and any per-day statement in the paper remain unchecked.

Additions not in the plan, decided after seeing results, none of which enters a verdict: `index_drift.csv`
(Table 8), `window_contributions.csv`, and four columns of `sizing_rules_33.csv` (rebalancing turnover, the
average absolute change of the log forecast, and gross and net utility of each of the 33 sizing rules). The test
ledger adapter (study code P103ECON) was planned.

## Limitations

* **The yardstick rests on assumptions.** A constant Sharpe ratio within the day is BHHP's assumption at a daily
  horizon, carried over to 30-minute windows; the fee's size depends on SR and gamma (Table 3), its p-value does
  not. The utility is not robust to noise in RV in Patton's sense (plan Section 3.8), but it ranks the models
  almost exactly as QLIKE does here.
* **Index for futures.** Basis, futures quotes, lot sizes and market impact are not modelled; the opening value
  and bar closes are not guaranteed prices; slippage is an assumption. As in P90 this most likely flatters every
  rule, which only strengthens the negative trading result.
* **One simple rule.** The rule resizes fully every 30 minutes. A no-trade band, a lower rebalancing frequency or
  an investor who already holds Nifty overnight would trade far less; the T2 row "rebalancing trades only" shows
  the ranking does not change, but such variants were not registered and were not run.
* **Same data as the headline.** The hybrid was chosen after the first full-sample run; this study prices the
  gain on the same 792 days and says nothing about new data. One seed set (P99 studies seed noise).
* **No option trade.** A straddle rule was out of scope: only end-of-day option prices are available, and the
  hybrid forecasts intraday windows, not the horizon an end-of-day straddle price covers.

## Files

* `experiments/p103_economic_value.py`: the script. It reuses P90's cost schedule, Sharpe, PSR, DSR, HAC and
  bootstrap functions (`experiments/f2_economics.py`) and the repository's DM test and Holm correction
  (`niftygex.evaluation`).
* `tests/test_p103_economic_value.py`: 19 unit tests on synthetic data (the utility formula and its maximum, the
  annualisation, the optimal forecast under noise, the non-overlap selection, positions and caps, turnover and
  exact cost accounting, the reused cost schedule, tracking loss, net utility, and a synthetic comparison in which
  the better forecast earns a positive fee and the test favours it).
* `results/p103_economic_value/`: `utility_tests.csv` (Tables 1 and 2), `fee_grid.csv` (Table 3),
  `qlike_same_windows.csv`, `utility_models.csv`, `daily_utility.csv` (each day's average bracket, from which Part
  A can be recomputed), `trading_metrics.csv` (Tables 4 and 5), `trading_tests.csv` (Table 6), `dsr.csv`
  (Table 7), `bootstrap_sharpe.csv`, `trading_daily.csv` (each day and strategy), `sizing_rules_33.csv`,
  `index_drift.csv` (Table 8), `window_contributions.csv`, `checks.csv`, `verdict.csv`, `run_info.json` and
  `stdout.txt`.

## Sources

1. Bollerslev, T., Hood, B., Huss, J. and Pedersen, L. H. (2018), "Risk Everywhere: Modeling and Managing
   Volatility", Review of Financial Studies 31(7), 2729-2773, doi:10.1093/rfs/hhy041. Details confirmed through
   search-engine summaries only (see the plan, Section 3.1).
2. Bailey, D. H. and Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio", Journal of Portfolio Management
   40(5), 94-107; (2012), "The Sharpe Ratio Efficient Frontier", Journal of Risk 15(2), 3-44 (as in P90).
3. Costs: `docs/prereg/P90_F2_ECONOMICS.md`, section 4 and its sources.
