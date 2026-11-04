# P103 pre-registration: is the hybrid's forecasting gain worth anything economically?

**Status.** Written on 3 November 2026 (phase 1) by an AI assistant (Claude) working on the project. **This
plan is committed to git before any outcome of this study has been computed.** No realized utility, no profit
and loss, no tracking error, no cost of any trading rule and no test statistic of this study has been computed.
What was looked at before writing it: the column names, row counts and forecast origins (minutes after 09:15)
of `results/pipeline_nse_v2/forecasts_{next15,next30,last_hour}.csv` (no realized value and no forecast value
was inspected), and the committed, already public QLIKE levels in `results/pipeline_nse_v2/scoreboard_*.csv`,
used only for the expectation in Section 1. The commit that adds this file is the time stamp; phase 2 (the
code, the run and `docs/P103_ECONOMIC_VALUE.md`) starts only after it has been pushed. All results will be
REAL DATA. **This is a research evaluation of the economic size of a statistical finding, not trading advice.**

## In plain words

The headline forecasting result says the hybrid (the anchor paper's mcsGARCH multiplied by a LightGBM
correction) forecasts Nifty's intraday volatility better than mcsGARCH: its QLIKE loss is 5.8%, 7.6% and 6.4%
lower at 15 minutes, 30 minutes and the last hour. A lower statistical loss does not say whether anyone would
pay for the better forecast. This study asks that in two ways.

* **Part A, the standard yardstick.** Imagine an investor who wants a steady level of risk. Before each window
  of 30 minutes she sets her Nifty position to "target risk divided by forecast volatility": a large position
  when the forecast says the market will be calm, a small one when it says it will be wild. If the forecast is
  right, her risk is exactly what she wanted; if it is wrong, she takes too much or too little risk. Bollerslev,
  Hood, Huss and Pedersen (2018) turned this into a number: the "realized utility" of a forecast, in percent of
  wealth per year. The difference between two forecasts is **the yearly fee the investor would pay to use the
  better one**. We compute that fee for the hybrid against mcsGARCH, and test whether it is above zero.
* **Part B, a real trading rule with costs.** The same idea as an actual rule: hold a long Nifty futures
  position over each 30-minute window, sized by the forecast, and pay the taxes, fees and slippage of every
  change in position (the cost schedule of study P90). Does the hybrid keep risk closer to the target, and is
  that worth the extra trading it causes?

We expect (Section 1) the fee to be clearly above zero statistically but tiny in money terms, and the trading
rule to lose money after costs whichever forecast it uses, because an intraday-only rule must buy and sell its
whole position every day. The plan says in advance how each outcome will be labelled.

## 1. Background and prior knowledge

All REAL DATA, from committed files (`results/pipeline_nse_v2/`, base seed 7; 792 out-of-sample test days from
10 August 2023 to 2 November 2026, 169 of them Nifty expiry days):

| Target | Forecast origins (minutes after 09:15) | hybrid_base vs mcsgarch, QLIKE | hybrid_base vs mcsgarch_iv, QLIKE |
|---|---|---|---|
| next 15 minutes | 0, 15, ..., 360 (25 per day) | -5.80% (DM p 1.8e-9) | -4.8% (p 0.005) |
| next 30 minutes | 0, 15, ..., 345 (24 per day, overlapping) | -7.65% (DM p 3.6e-15) | -5.9% (p below 0.001) |
| last hour | 315 (14:30, one per day) | -6.36% (DM p 0.00010) | not tested here |

(`scoreboard_*.csv`, `tests_*.csv`, `docs/REFEREE_RESPONSE_FORECAST.md`.) Against `mcsgarch_iv`, the mcsGARCH
whose daily level is the previous evening's straddle-implied variance, the edge comes from the weekly-expiry
periods since November 2024 (about 0.2% and 0.7% before, 7% to 9% after; Giacomini and White tests in
`docs/REFEREE_RESPONSE_FORECAST.md`). The QLIKE levels at 30 minutes are 0.2176 (mcsgarch), 0.2135
(mcsgarch_iv) and 0.2009 (hybrid_base) per forecast.

**Expected size (written before any utility was computed).** The utility loss of a forecast (Section 3.1) is
(1/2)(1 - sqrt(RV/F))^2. For RV/F near 1 this is about one quarter of QLIKE, and for very large RV/F about one
half. With the published 30-minute QLIKE gap of 0.0166 per forecast and the primary fee scale SR^2/gamma = 0.08
per year (Section 3.2), the hybrid's fee over mcsGARCH should therefore be roughly 0.08 x 0.0166 / 4 = 0.03% to
0.08 x 0.0166 / 2 = 0.07% of wealth a year, that is about Rs 3,000 to Rs 7,000 a year per Rs 1 crore. This is
an approximation on all 24 origins, not a result. **We expect H1 to pass and the size label (Section 5) to be
"tiny".** We expect the standalone trading rule to fail its Deflated Sharpe Ratio test (power note, Section 4.6).

## 2. Data and sample

* **Sample.** The 792 test days of the v2 run (`results/pipeline_nse_v2/forecasts_*.csv`), all of them, with
  their `is_expiry` and `regime` labels. Subsets (from `niftygex.evaluation.subgroups`, as in the pipeline):
  expiry and non-expiry days, the 729 days before NSE's closing auction (3 August 2026), the development window
  (to 30 June 2025) and the hold-out (from 1 July 2025), and the three expiry regimes (312, 192 and 288 days).
* **Forecasts.** Column `y` is the realized variance over the window (sum of squared 1-minute log returns over
  minutes [m, m + h) after 09:15, the first minute's return measured from the 09:15 bar's own open, so the
  overnight gap is excluded). Model columns are variance forecasts made with information up to minute m.
* **Returns for Part B.** `data/raw/nifty_1min.csv` (git-ignored; listed in `data/MANIFEST.csv`), read with
  `niftygex.data.load_minute_bars` and `to_session_grid`, the functions the pipeline uses. A window's return is
  exp(sum of the same 1-minute log returns) - 1, so it is exactly the return whose squared pieces make `y`.
  **Stop rule:** the script rebuilds `y` from the bars for every window it uses and stops if any differs from
  the file by more than 1e-9 in relative terms.
* **Other checks that stop the script:** every day has its full set of origins (25, 24 and 1); every forecast
  used is finite and positive; the QLIKE margins of hybrid_base against mcsgarch on all origins reproduce the
  published -5.80%, -7.65% and -6.36% to 1e-9 (absolute, in percent).

## 3. Part A: realized utility

### 3.1 The investor and the formula

The framework is that of **Bollerslev, Hood, Huss and Pedersen (2018), "Risk Everywhere: Modeling and Managing
Volatility", Review of Financial Studies 31(7), 2729-2773, doi:10.1093/rfs/hhy041** (BHHP). An investor with
mean-variance preferences holds a fraction x of wealth in a risky asset whose conditional Sharpe ratio SR is
constant, so its expected excess return over a period is SR x sigma, with sigma the conditional volatility. Her
utility for one period is

    U = x SR sigma - (gamma / 2) x^2 sigma^2,          gamma = relative risk aversion.

With a variance forecast F she holds x = SR / (gamma sqrt(F)), the optimal position if F = sigma^2, so she
targets an annualised volatility of SR / gamma. Replacing sigma^2 ex post by the realized variance RV of the
period gives the **realized utility**

    u(F, RV) = (SR^2 / gamma) x [ sqrt(RV / F) - RV / (2 F) ].

The bracket b = sqrt(RV/F) - RV/(2F) is largest, 1/2, when F = RV, so a perfect forecast earns SR^2 / (2 gamma).
We write the shortfall as the **utility loss** l = 1/2 - b = (1/2)(1 - sqrt(RV/F))^2 >= 0.

**What was and was not verified.** Direct downloads of the paper (the Duke, CBS and Aarhus copies) and of later
papers that use it were blocked by this session's network settings, so the paper's text was not read. Through
search-engine summaries of the paper and of later papers that apply it, we confirmed: the journal, volume,
issue, pages and DOI above; that the framework assumes a mean-variance investor and a constant Sharpe ratio;
that realized utility replaces the expectation by realized volatility; that BHHP use **SR = 0.4 (annualised)
and gamma = 2**, so a 20% volatility target; that a perfect forecast earns SR^2/(2 gamma) (4% a year with those
values); and that realized utility is read as the percentage of wealth the investor would pay for the model's
forecasts. **The formula above is our derivation from those assumptions; its exact typography and equation
number in the paper were not checked**, nor whether BHHP state it per day and then annualise. If the paper's
form differs, that is reported as a deviation and both versions are shown.

**Our adaptation.** BHHP work with daily forecasts. Here a "period" is one intraday window (15 minutes, 30
minutes or the last hour) and RV is `y`. We assume the constant Sharpe ratio holds within each window. If there
are n windows a year and the annual Sharpe ratio is SR_a, the per-window Sharpe ratio is SR_a / sqrt(n), so the
yearly utility is n x (SR_a^2 / n / gamma) x mean(b) = (SR_a^2 / gamma) x mean(b), **whatever the window
length**. The investor is invested only during the windows used (no overnight position).

### 3.2 The fee

For a model M, its annualised realized utility in percent of wealth is

    U_M = 100 x (SR^2 / gamma) x mean over windows of b(F_M, RV).

**Fee of the hybrid over a benchmark** = U_hybrid - U_benchmark, in percentage points of wealth per year: the
yearly fee the investor would pay to switch from the benchmark's forecasts to the hybrid's. Also given in
rupees per year per Rs 1 crore of wealth (fee / 100 x 10,000,000), and as the share of the benchmark's
shortfall from a perfect forecast that the hybrid closes, (U_hybrid - U_benchmark) / (U_max - U_benchmark),
with U_max = 100 x SR^2 / (2 gamma).

* **Primary values: SR = 0.4 (annualised), gamma = 2** (BHHP's values as confirmed above), so SR^2/gamma = 0.08
  and U_max = 4% a year.
* **Sensitivity grid** (fee only): SR in {0.2, 0.4, 0.6, 0.8} x gamma in {1, 2, 5, 10}. The fee is proportional
  to SR^2/gamma, so the grid rescales it; **every p-value in Part A is the same for every SR and gamma**,
  because the tests are on the bracket b.

### 3.3 Windows (non-overlapping)

* **30 minutes (primary): origins 0, 30, 60, ..., 330**, 12 windows a day covering 09:15 to 15:15 without
  overlap. Fixed now; the alternative offset (15, 45, ..., 345) is a robustness check.
* **15 minutes: all 25 origins 0, 15, ..., 360** (09:15 to 15:30), which do not overlap.
* **Last hour: origin 315** (14:30 to 15:30), one window a day.

### 3.4 Test

For each day t and model M, b_t(M) is the average of b over the day's windows (the day is the unit, as in every
DM test of the project). For "the hybrid gives higher average realized utility than benchmark B":
`niftygex.evaluation.diebold_mariano(l_t(B), l_t(hybrid_base), alternative="greater")`, with l_t = 1/2 - b_t
the daily utility loss: the repository's DM test (Newey-West variance with its default lag,
floor(4 (n/100)^(2/9)) = 6 for 792 days, the Harvey, Leybourne and Newbold correction, Student t with n - 1
degrees of freedom), **one-sided**. Also reported: the mean daily difference, the DM statistic, and the share of
days on which the hybrid's utility is higher.

### 3.5 Primary hypothesis

**H1. At 30 minutes, on all 792 test days, with the 12 non-overlapping windows a day (origins 0, 30, ..., 330),
hybrid_base gives a higher average realized utility than mcsgarch. Supported if the one-sided p of Section 3.4
is below 0.05.**

### 3.6 Secondary hypotheses (Holm over these four)

* **S1.** Same as H1 at 15 minutes (25 windows a day), hybrid_base vs mcsgarch.
* **S2.** Same in the last hour (one window a day), hybrid_base vs mcsgarch.
* **S3.** 15 minutes, hybrid_base vs **mcsgarch_iv** (information-matched benchmark).
* **S4.** 30 minutes, primary windows, hybrid_base vs **mcsgarch_iv**.

Holm-adjusted p-values (`niftygex.evaluation.holm`) over S1 to S4; each is supported if its Holm p is below
0.05. They cannot change the verdict on H1.

### 3.7 Robustness (registered, reported, no decision)

* **R1.** 30 minutes with the other non-overlapping offset, origins 15, 45, ..., 345.
* **R2.** 30 minutes with all 24 (overlapping) origins, averaged within the day.
* **R3.** 30 minutes, primary windows without the 09:15 window (origins 30, ..., 330), because a large part of
  the QLIKE gain is at the open.
* **R4.** H1 and S4 on each subset of Section 2 (expiry, non-expiry, pre-auction, development, hold-out, each
  regime), the S4 regime split answering the "only after November 2024" point.
* **R5.** The annualised utility of every one of the 32 models in the forecast files at each horizon and its
  fee over mcsgarch (descriptive, no p-values).
* **R6.** The fee grid of Section 3.2, and the QLIKE margin on exactly the windows of H1 and S1 to S4, so the
  reader can compare the two yardsticks.

### 3.8 A property of this utility that the reader should know

Unlike QLIKE, this utility is not "robust" in Patton's (2011) sense when RV is a noisy measure of the true
variance: maximising the expected bracket over F gives F = (E RV)^2 / (E sqrt(RV))^2, which is at least E RV,
so a forecast that is slightly too high is rewarded (our derivation, by Jensen's inequality). For the investor
this is not a flaw, since RV is the risk she actually bears; but it means the utility and QLIKE can rank two
models differently. Phase 2 reports both on the same windows (R6).

## 4. Part B: a volatility-targeted trading rule after costs

### 4.1 The rule

* **Windows**: the 12 non-overlapping 30-minute windows of H1 (origins 0, 30, ..., 330), on all 792 test days.
* **Position** (units of notional per unit of capital, long only): w_k = min(cap, sigma*_w / sqrt(F_k)), F_k the
  model's 30-minute variance forecast for window k, **target sigma* = 20% a year = SR/gamma** (the investor of
  Part A), sigma*_w = 0.20 / sqrt(252 x 12) per window (252 days x 12 windows a year), **leverage cap = 4**. A
  10% target with a cap of 2 gives exactly half of every position, profit, cost and utility number below, and
  the same Sharpe ratios, tracking errors and p-values.
* **Strategies (the registered rule variants, N = 4):** sized by `hybrid_base`, by `mcsgarch`, by
  `mcsgarch_iv`, and a **constant** position w = 2 (no forecast: the position that would hit the target if
  Nifty's intraday volatility were 10% a year; an assumption, not estimated).
* **Timing.** The position for window k is set at its start (the 09:15 bar's open for the first window, the
  close of minute 30k - 1 otherwise) using only the forecast made at that origin, and held to the close of
  minute 30k + 29. The rule is flat outside the windows: it opens at 09:15, closes at 15:15 (the close of the
  15:14 bar) and holds nothing from 15:15 to 15:30 or overnight. No compounding within the day.
* **Profit and loss** (bp of capital): day t gross = 10,000 x sum over k of w_k R_k, R_k the window's index
  return (Section 2).
* **Instrument assumption (disclosed).** Index minute bars stand in for near-month Nifty futures: basis
  changes, futures quotes and the lead of futures over the index are not modelled, a bar close is one last
  trade and not a price anyone was sure to get, and positions are continuous (no rounding to whole lots, which
  matters little for an account of many lots and a lot for a small one). Entering at the 09:15 bar's open
  assumes the index opening value was tradable. As in P90, this most likely flatters every rule.

### 4.2 Costs

The cost schedule of P90 (`docs/prereg/P90_F2_ECONOMICS.md`, section 4), reused from `cost_components` in
`experiments/f2_economics.py`: securities transaction tax on the sell leg (1.25 bp to 30 September 2024, 2 bp
to 31 March 2026, 5 bp from 1 April 2026), exchange and SEBI charges, stamp duty on the buy leg, brokerage of
Rs 20 per order converted at Rs 15 lakh of notional (charged pro rata to the amount traded), GST, and
**slippage of 1 bp per side (base case)**. The round-trip cost per unit of notional is C(date) = 4.26, 4.97 and
7.97 bp in the three periods (base case).

Day t's trades: from 0 to w_1 at 09:15, from w_(k-1) to w_k at each later window start, and from w_12 to 0 at
15:15. **Turnover** tau_t = |w_1| + sum over k = 2 to 12 of |w_k - w_(k-1)| + |w_12|. Because the rule starts
and ends each day flat, the units bought equal the units sold (tau_t / 2 each), so the day's cost is exactly
**c_t = (tau_t / 2) x C(date)** bp of capital, whichever leg carries which charge. Net = gross - c_t. Reported
split: the opening and closing trades (|w_1| + |w_12|) and the rebalancing trades. **Stress grid:** gross (no
cost), and slippage 0, 0.5, 1, 2 and 3 bp per side with statutory costs by date (as P90). Cost levels are not
trials.

### 4.3 Metrics (each strategy, each cost level unless stated)

* **Tracking (the purpose of volatility targeting; cost-free by construction).** Day t's realized strategy
  volatility, annualised: s_t = sqrt(252 x sum over k of w_k^2 y_k). **Tracking loss TL_t = (ln(s_t / sigma*))^2**,
  and the root mean squared log deviation RMSLD = sqrt(mean TL_t). Also: realized volatility over the whole
  test period, both sqrt(252 x mean over days of sum_k w_k^2 y_k) and the standard deviation of the window
  returns w_k R_k times sqrt(3024), each divided by the target; the share of days with s_t between 0.8 and
  1.25 times the target; the mean position and the share of windows at the cap.
* **Returns.** Mean and median net bp per day, annualised mean (x 252), annualised volatility of the daily net
  series, **annualised Sharpe ratio** (daily mean / daily standard deviation x sqrt(252); no risk-free rate,
  since a futures position needs no cash), HAC t of the daily mean (P90's `hac_t`, Newey-West, 5 lags, one-sided
  p for mean > 0), maximum drawdown, turnover per day, cost per year in percent of capital.
* **Net realized utility** (the Part A investor running this rule; constant-Sharpe expected return, so it does
  not depend on Nifty's realized intraday drift): U_t = sum over k of [w_k SR_w sqrt(y_k) - (gamma/2) w_k^2 y_k]
  - c_t / 10,000, with SR_w = 0.4 / sqrt(3024) and gamma = 2; annualised as 100 x 252 x mean(U_t) percent of
  wealth. Also gross (c_t = 0), and with the realized return w_k R_k in place of w_k SR_w sqrt(y_k)
  (descriptive).
* **Probabilistic and Deflated Sharpe Ratio** (P90's `psr` and `dsr`; Bailey and Lopez de Prado 2012, 2014) on
  each strategy's daily net series at base-case costs, T = 792: DSR = PSR(SR_0), SR_0 = sqrt(V) x
  ((1 - g) InvPhi(1 - 1/N) + g InvPhi(1 - 1/(N e))), g = 0.5772. **N = 4** (the four registered strategies;
  cost levels are not trials), and **V = the sample variance (ddof = 1) of the four strategies' daily net Sharpe
  ratios at base-case costs.** Sensitivities (no decision): N = 33 (each of the 32 forecasting models in the
  files used as a sizing rule, plus the constant rule; V from those 33), and V = 1/T (the value for pure
  noise) with N = 4 and 33. N = 4 is honest for this study only: it does not count the 31 challengers from which
  the hybrid was chosen; the N = 33 sensitivity does.
* **Bootstrap.** Stationary bootstrap (Politis and Romano 1994; P90's `stationary_bootstrap_indices`), mean
  block 5 days, 10,000 draws, seeds `numpy.random.default_rng([20261102, 103, k])`, k = 0 (hybrid_base),
  1 (mcsgarch), 2 (mcsgarch_iv), 3 (constant), 4 (paired draws for the difference): 95% percentile interval of
  each annualised Sharpe ratio at base-case costs, and of the difference hybrid minus mcsgarch (the same day
  draws for both; reported as an interval and the share of draws at or below zero, not as a test).

### 4.4 Registered trading tests (hybrid_base against mcsgarch)

* **T1 (tracking).** `diebold_mariano(TL_t(mcsgarch), TL_t(hybrid_base), alternative="greater")`, one-sided:
  the hybrid's daily tracking loss is lower.
* **T2 (net utility, base-case costs).** `diebold_mariano(-U_t(mcsgarch), -U_t(hybrid_base),
  alternative="greater")`, one-sided: the hybrid's daily net realized utility is higher, after paying for its
  own trades.
* Reported with them (robustness, no decision): T1 and T2 against mcsgarch_iv; T2 at every cost level; T2 with
  only the rebalancing trades charged (as for an investor who holds Nifty anyway and only adjusts the size
  intraday).

### 4.5 Out of scope: a straddle rule

A rule that sells or buys option straddles when the hybrid's forecast is below or above the implied variance is
**not part of this study**. NSE intraday option prices are not available to the project (only the end-of-day
bhavcopy), and the hybrid forecasts intraday windows of 15 minutes to an hour, not the overnight-to-expiry
horizon that an end-of-day straddle price covers. Any such rule would need data we do not have.

### 4.6 Power note (before the run)

* **Standalone DSR.** With T = 792 days and N = 4, a strategy needs an estimated annualised Sharpe ratio above
  about **0.93** (if V is near 0) to **1.52** (if V = 1/T) for DSR > 0.95 (normal returns; `f2_dsr_power`
  arithmetic, no data). With N = 33 the bar rises to about 2.12 (V = 1/T). Every rule here opens and closes its
  whole position every day, and the forecast rules also follow the intraday volatility pattern (small at the
  open, large at lunch, smaller again towards the close). We guess a turnover of about 4 to 8 units a day, so
  base-case costs take an annualised Sharpe ratio of roughly 1.25 to 2.5 in the 4.97 bp period and 2.0 to 4.0
  from April 2026 (at a 20% target a day's volatility is about 126 bp, and the drag is
  (turnover / 2) x cost / 126 bp x sqrt(252)). The gross Sharpe ratio would therefore need to be about 2.2 to 4. Whether Nifty earns anything at all during
  the trading day (rather than overnight) is not known to us for this period; for other markets much of the
  equity premium has been found overnight (Lou, Polk and Skouras 2019, cited from memory, not looked up). **We
  therefore expect the standalone DSR test to fail for every strategy, almost regardless of the forecast; it
  measures the cost of trading intraday, not the value of the hybrid.** That is why the decision on the
  hybrid's value rests on Part A and on T1 and T2, which compare forecasts within the same rule.
* **T2.** If the utility fee is near the 0.03% to 0.07% a year expected in Section 1, a difference in turnover
  of only 0.01 units a day at about 5 bp per round trip (0.005 x 5 bp x 252 days, about 0.06% a year) is the
  same order of magnitude as the gain, so T2 may go either way and is likely to have low power.

## 5. Decision rules (fixed before any outcome)

**Part A.** H1 supported (p < 0.05): "the hybrid's 30-minute forecasting gain has positive economic value for a
volatility-timing investor". Otherwise: "no economic value shown at 30 minutes". **Size label**, from the fee
of hybrid_base over mcsgarch at 30 minutes at SR = 0.4, gamma = 2: **below 0.10% of wealth a year: "tiny"; 0.10%
to below 0.50%: "small"; 0.50% or more: "material"** (round numbers chosen in advance, not taken from the
literature; 0.10% a year is Rs 10,000 a year per Rs 1 crore). The label is stated with the rupee amount in the
first paragraph of the write-up, whatever the p-value.

**Part B.** "**Survives implementation**" if T1 p < 0.05 and T2 p < 0.05. "**Better risk control, not worth
its trading costs**" if T1 p < 0.05 and T2 p >= 0.05. "**No implementation gain**" if T1 p >= 0.05.
**Standalone:** the hybrid's rule is "a standalone economically meaningful strategy" if its DSR (N = 4,
base-case costs) is above 0.95; otherwise "not a standalone strategy", with the power note above.

**Overall verdict** (first paragraph of `docs/P103_ECONOMIC_VALUE.md`):

* **"Economically meaningful"** if H1 is supported, the size label is "small" or "material", and Part B is
  "survives implementation".
* **"Detectable but not economically meaningful"** if H1 is supported and either the label is "tiny" or Part B
  is not "survives implementation"; the write-up says which.
* **"No economic value shown"** if H1 is not supported.

S1 to S4, the robustness checks, the standalone DSR and the cost grid cannot change the overall verdict. If S4
(against the information-matched mcsgarch_iv at 30 minutes) fails while H1 passes, the write-up says in its
first paragraph that the economic value over mcsGARCH is not shown to come from the method rather than from the
implied-variance information.

## 6. Deviations policy

* After this file is committed the design does not change. Any change is a deviation, listed in
  `docs/P103_ECONOMIC_VALUE.md` with its reason and whether any outcome had been seen when it was decided.
* If one of the stop rules of Section 2 fires, the cause is found and reported; the run continues only if the
  fix changes no number fixed here (for example, a reading error).
* Points this plan does not settle are decided in phase 2 without looking at outcomes where possible, and are
  listed as "points where the plan was silent".

## 7. Known weaknesses

1. **Constant Sharpe ratio within the day** is an assumption (BHHP's, at a daily horizon); intraday it may not
   hold, and the Part A fee depends on the chosen SR and gamma (the tests do not).
2. **The utility is not a robust loss** (Section 3.8): it can favour a slightly too high forecast.
3. **Index for futures, continuous positions, pro rata brokerage, assumed slippage**: see Section 4.1 and P90.
4. **Intraday only.** The rule pays a full round trip every day; an investor who holds Nifty anyway would pay
   only the rebalancing trades (reported, Section 4.4).
5. **Same data as the headline.** The hybrid was chosen after the first full-sample run
   (`docs/REFEREE_RESPONSE_FORECAST.md`, M4.2); this study prices the gain, it does not test it on new data.
6. **One seed set.** Forecasts are the headline run's (base seed 7); seed noise is the subject of P99.

## 8. Code and outputs (phase 2)

* `experiments/p103_economic_value.py` writes `results/p103_economic_value/`: `utility_tests.csv` (H1, S1 to
  S4 with Holm, R1 to R4), `utility_models.csv` (R5), `fee_grid.csv` and `qlike_same_windows.csv` (R6),
  `daily_utility.csv` (each day's average bracket for the main models at each horizon and window set),
  `trading_metrics.csv` (strategy x cost level), `trading_tests.csv` (T1, T2 and their robustness versions),
  `trading_daily.csv` (each day and strategy: gross, turnover, cost, net, s_t, U_t), `dsr.csv`,
  `bootstrap_sharpe.csv`, `checks.csv` (the stop rules), `verdict.csv`, `run_info.json` (this plan's commit,
  seeds, input file hashes, versions) and `stdout.txt`.
* `tests/test_p103_economic_value.py` (synthetic data only): the utility formula and its maximum, the
  non-overlap selection, the cost accounting (buys equal sells, cost = turnover / 2 x round trip), the tracking
  loss, position caps, and sanity checks on simulated forecasts (a perfect forecast earns SR^2/(2 gamma); a
  noisier forecast earns less, and the test favours the better forecast).
* `docs/P103_ECONOMIC_VALUE.md`, the study added to `scripts/reproduce_all.py`, and a ledger adapter (study code
  P103ECON) in `scripts/build_test_ledger.py`.

## Sources

1. Bollerslev, T., Hood, B., Huss, J. and Pedersen, L. H. (2018), "Risk Everywhere: Modeling and Managing
   Volatility", Review of Financial Studies 31(7), 2729-2773, doi:10.1093/rfs/hhy041. Abstract page:
   https://academic.oup.com/rfs/article-abstract/31/7/2729/5001472 ; SSRN: https://papers.ssrn.com/abstract=3122329 .
   Details confirmed through search-engine summaries only (Section 3.1); the full text could not be fetched.
2. Bailey, D. H. and Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio", Journal of Portfolio Management
   40(5), 94-107, and (2012), "The Sharpe Ratio Efficient Frontier", Journal of Risk 15(2), 3-44: as cited and
   implemented in P90 (`docs/prereg/P90_F2_ECONOMICS.md`, sources 13 and 14).
3. Cost sources: P90 pre-registration, section 4 and sources 1 to 12.
4. Cited from memory, not looked up: Patton, A. J. (2011), "Volatility forecast comparison using imperfect
   volatility proxies", Journal of Econometrics 160(1); Politis, D. N. and Romano, J. P. (1994), "The Stationary
   Bootstrap", JASA 89(428); Lou, D., Polk, C. and Skouras, S. (2019), "A tug of war: Overnight versus intraday
   expected returns", Journal of Financial Economics 134(1).
