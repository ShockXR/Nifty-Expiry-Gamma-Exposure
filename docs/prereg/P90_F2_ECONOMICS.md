# P90 pre-registration: how big is F2 after realistic trading costs?

Written 2 November 2026, before any return data were loaded or any backtest was run for this study, and
committed to git before phase 2. Later changes are listed as deviations in `docs/F2_ECONOMICS.md`. **This is a
research evaluation of the economic size of a statistical finding, not trading advice.**

## 1. Background, and what is already known

F2 (`docs/REAL_DATA_RESULTS.md` R5, `docs/FALSIFICATION.md`): Nifty's last-30-minute return (14:59 to 15:29
bar closes) tends to reverse the "rest of day" return (previous day's 15:29 bar close to today's 14:59 bar
close), only after evenings when GEX under NSE's "Pro" participant rule was negative (interaction -0.068, HAC
p = 0.0002, 929 days). A clear pattern can still be too small to pay for trading. Known already: before costs,
trading against the day on negative-Pro-GEX days earned about +2.1 bp per trade over the full sample (the
"follow" payoff was -2.1 bp, t = -2.3), less than the base-case cost below (about 5 bp per round trip, 8 bp
from April 2026). **We therefore expect the primary rule to fail after costs**; the study measures by how
much, and whether trading only on large moves does better.

## 2. Trading rules (fixed now)

**Inputs** (no new data): `results/pipeline_nse/daily_market.csv`, `results/pipeline_nse/daily_gex_panel.csv`,
`data/raw/nifty_1min.csv`, read with `daily_returns` and `RULES["participant_pro"]` from
`experiments/momentum_gex.py`. On day t: `neg_t` = 1 if `net_gex_participant` on the previous trading day is
below 0 (built from files NSE publishes after that close, so known on day t), else 0; no trade if GEX is
missing. `r_rod_t` = log(14:59 bar close / previous day's 15:29 bar close) in bp. Trade return
R_t = (exit price / entry price - 1) x 10,000 bp, from index bar closes. Position w_t is in units of notional
(+1 long, -1 short, 0 flat); profit and loss is in bp of notional, with no compounding or leverage.

* **Primary rule P.** At the 14:59 bar close, if `neg_t` = 1, take w_t = -sign(r_rod_t) (sell after an up
  day, buy after a down day). Exit at the 15:29 bar close. Flat otherwise. No parameters.
* **Secondary S1 (sized by the move).** Same days, w_t = clip(-r_rod_t / s, -3, 3), with s the median
  |r_rod| over negative-GEX days in the training window.
* **Secondary S2 (trade only when the forecast beats costs).** Fit R5's binary regression
  r_last = a + b r_rod + c (r_rod x neg) + d neg by OLS on the training window. On a day with `neg_t` = 1,
  the forecast gain is g_t = -sign(r_rod_t) x (a + d + (b + c) r_rod_t). Take w_t = -sign(r_rod_t) if
  g_t > C_t (the round-trip cost on that date, section 4, at the cost level being evaluated), else flat.
* **Sensitivity D (not a decision input).** Rule P entered at the 15:00 bar close, since nobody can trade at
  the exact close of the bar that produces the signal.

**Walk-forward.** Training period 2 Jan 2023 to 31 Dec 2024 (the split of FALSIFICATION T4). Test period: first
trading day of 2025 to 31 Jul 2026, in consecutive blocks of 21 trading days. Before each block, s (S1) and
a, b, c, d (S2) are re-estimated on all days from 2 Jan 2023 to the day before the block (expanding window).
P and D have nothing to fit. 2023 to 2024 results are shown as in-sample context only. The test period is not
truly unseen (R5 was found on the full sample), which is why the decision uses the Deflated Sharpe Ratio.

**Closing-auction days are excluded** from the test period and every decision metric: from 3 Aug 2026 the
index is frozen 15:16 to 15:28 and the 15:29 bar is the auction price, which a futures trader cannot exit at.
These 63 days are reported separately, labelled "not tradable as measured".

## 3. Instrument assumption

Index minute bars stand in for near-month Nifty futures (on a monthly futures expiry day, the next-month
contract, because the expiring one settles at the official close). Basis, futures quotes and liquidity are not
modelled. Carry moves the basis by well under 0.1 bp in 30 minutes; intraday basis changes and futures leading
the index add noise of unknown sign; and a bar close is one last trade, not a price anyone was sure to get.
Overall the proxy is more likely to flatter the rule than hurt it, so slippage is charged on top of statutory
costs and stressed up to 3 bp per side.

## 4. Costs per round trip, in bp of notional

A round trip is one buy and one sell. Percentage charges use the entry notional on both legs (error under
0.05 bp), at the rate in force on the trade date. 1 bp = 0.01%; 1 crore = Rs 10 million; 1 lakh = Rs 100,000.

| Component | Rate | Charged on | bp per round trip | Source |
|---|---|---|---|---|
| Securities Transaction Tax (STT), futures | 0.01% to 31 Mar 2023; 0.0125% 1 Apr 2023 to 30 Sep 2024; 0.02% 1 Oct 2024 to 31 Mar 2026; 0.05% from 1 Apr 2026 | sell leg | 1.00; 1.25; 2.00; 5.00 | [1] to [4] |
| NSE transaction charge incl. IPFT | about 0.0020% per side to 30 Sep 2024 (0.00198% to 0.00201% by sub-period); 0.00183% from 1 Oct 2024 (Rs 173 + Rs 10 IPFT per crore; total unchanged by the March 2026 IPFT change) | both legs | 0.400; 0.366 | [5] to [8] |
| SEBI turnover fee | 0.0001% (Rs 10 per crore) | both legs | 0.020 | [9] |
| Stamp duty | 0.002% | buy leg | 0.200 | [10] |
| Brokerage | Rs 20 per order (discount broker; 0.03% cap does not bind) at a notional of Rs 15 lakh, SEBI's minimum index-derivative contract value from 20 Nov 2024 | both legs | 0.267 | [9] [11] |
| GST | 18% of brokerage + exchange charge + SEBI fee | | 0.124; 0.117 | [9] |
| Slippage (**assumption**) | 1 bp per side: half the near-month futures spread (well under 0.5 bp) plus small-order impact and the gap between a bar close and an obtainable price | both legs | 2.00 | [12] |

**Base-case total per round trip:** 4.01 bp (to 31 Mar 2023), 4.26 bp (1 Apr 2023 to 30 Sep 2024),
**4.97 bp (1 Oct 2024 to 31 Mar 2026)**, **7.97 bp (from 1 Apr 2026)**; STT is the largest part. Without
slippage: 2.01, 2.26, 2.97, 5.97 bp. **Stress grid:** slippage 0, 0.5, 1, 2 and 3 bp per side, statutory
costs unchanged, plus gross. For S1 costs are multiplied by |w_t|.

**Access note.** Sources consulted 2 Nov 2026. Direct fetches of zerodha.com and www.nseindia.com were blocked by
this session's network settings, so rates were read from search-engine summaries of the pages listed. No
authoritative published spread for Nifty futures was found. If a direct check finds a different statutory
rate, the rate above stays in the decision and the corrected result is shown next to it.

## 5. Metrics and exact formulas

Test period, base-case costs unless stated, repeated for each stress level. The daily series
net_t = w_t R_t - |w_t| C_t covers every test day (0 when flat); T is its length.

* **Counts and averages:** trades; mean and median net bp per trade and per day; hit rate = share of trades
  with net > 0.
* **Sharpe ratio:** SR = mean(net_t) / sd(net_t) per day (ddof = 1); annualised SR_a = SR x sqrt(252) (NSE has
  about 245 to 250 sessions a year, so this overstates SR_a by about 1%). No risk-free rate is subtracted: a
  futures position needs no cash outlay.
* **HAC t:** mean(net_t) / sqrt(LRV / T), LRV the Newey-West long-run variance (Bartlett kernel, 5 lags, as in
  R5); one-sided p for mean > 0.
* **Maximum drawdown:** with CUM_t the cumulative sum of net_t, MDD = max over t of (max over u <= t of CUM_u
  minus CUM_t), in bp of notional, plus its length in days.
* **Stationary bootstrap** (Politis and Romano 1994) of net_t: mean block 5 days, 10,000 resamples, 95%
  percentile interval for SR_a.
* **Break-even cost per round trip:** c* = sum(w_t R_t) / sum(|w_t|) (for S2 with the base-case trades held
  fixed). Break-even slippage per side = (c* - mean statutory and brokerage cost per unit traded) / 2.
* **Probabilistic Sharpe Ratio** [14]:
  `PSR(SR*) = Phi( (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + ((g4 - 1) / 4) SR^2) )`,
  SR per day (not annualised), g3 and g4 the skewness and kurtosis (3 for a normal, not excess) of net_t as
  moment estimators, Phi the standard normal distribution function. Reported: PSR(0).
* **Deflated Sharpe Ratio** [13]: `DSR = PSR(SR_0)`,
  `SR_0 = sqrt(V) x ( (1 - gamma) InvPhi(1 - 1/N) + gamma InvPhi(1 - 1/(N e)) )`, InvPhi the inverse of Phi,
  gamma = 0.5772 (Euler-Mascheroni), e = 2.71828. SR_0 is the Sharpe ratio the best of N worthless strategies
  would be expected to show.
  * **N = 824 trials** = 78 GEX tests of FALSIFICATION family A (42 for F1, 36 for F2; the 36 are R5's
    interaction tests, so R5 is not counted twice) + 742 tests of family B + 4 rules here (P, S1, S2, D).
    Cost levels are not trials. Sensitivities: N = 82 (family A and this study) and N = 3,624 (adding T3's
    2,800 random placebo sign rules). Other project studies (DEALER_SIGN, INTRADAY_GAMMA and others) are not
    counted, so 824 is a lower bound for the project. At N = 824 the multiplier of sqrt(V) is 3.20.
  * **V, variance of Sharpe ratios across trials.** The trials are regressions on different samples, so trial
    n's Sharpe ratio is taken as z_n / sqrt(T), T the length of the evaluated series. Family A:
    z_n = sign(coef) x InvPhi(1 - p/2) from `results/falsification/t7_family_A_original_tests.csv`; the 4
    rules here: z_n = HAC t of the base-case daily net mean. V = sample variance (ddof = 1) of these 82 values,
    used for every N, because family B mixes statistic types (permutation counts, Diebold-Mariano, IV) that
    cannot be put on this scale. This scaling is our adaptation, not the paper's. Reported: V, sqrt(V T)
    (the spread of the trials' t-statistics), SR_0 annualised.

## 6. Primary question and decision rule

**Question.** After realistic costs, does trading F2 earn a positive risk-adjusted return in the test period
that is too large to be explained by the number of things the project tried?

**Decision** (rule P, test period, base-case costs, daily series, N = 824):

* **"Economically meaningful"** if DSR > 0.95.
* **"Not economically meaningful"** if DSR <= 0.95, described as **"no net edge"** if the mean net return per
  day is <= 0 (break-even cost below base-case cost), and as **"positive but not distinguishable from selection
  and noise"** if it is above 0.

S1, S2 and D get the same metrics but cannot change the verdict; if one has DSR > 0.95 while P does not, it is
reported as a lead for a new pre-registered test on data after 2 Nov 2026. With about 90 trades expected
(negative-Pro-GEX days before the auction in 2025 onward, FALSIFICATION T4), power is low: "not meaningful"
means "not shown to be meaningful", and the break-even cost and stress grid give the economic size directly.

## 7. Reproducibility and future outputs

* Bootstrap seeds: `numpy.random.default_rng([20261102, 90, k])`, k = 0 (P), 1 (S1), 2 (S2), 3 (D);
  20261102 is the project's master seed.
* The script first re-estimates R5's Pro binary interaction and stops if it differs from
  `results/momentum_gex/regressions.csv` by more than 1e-8; input SHA-256 prefixes go in `run_info.json`.
* Script `experiments/f2_economics.py`; tables in `results/f2_economics/` (`cost_schedule.csv`, `trades.csv`,
  `daily_pnl.csv`, `walkforward_params.csv`, `metrics.csv`, `cost_grid.csv`, `bootstrap_sharpe.csv`,
  `dsr.csv`, `auction_days.csv`, `run_info.json`); log `results/f2_economics_stdout.txt`; synthetic-data
  tests `tests/test_f2_economics.py`; write-up `docs/F2_ECONOMICS.md`.

## Sources (accessed 2 Nov 2026 through search-engine summaries; see section 4)

1. NSE circular, STT from 1 Apr 2023 (futures 0.01% to 0.0125%): https://archives.nseindia.com/content/circulars/FATAX56235.pdf
2. MSEI circular 16075 (Finance (No. 2) Act 2024, futures 0.02% from 1 Oct 2024): https://msei.in/SX-Content/Circulars/2024/October/Circular-16075.pdf ; Business Standard, 1 Oct 2024: https://www.business-standard.com/markets/news/higher-stt-and-transaction-charges-on-nse-bse-mcx-effective-today-124100100146_1.html
3. Zerodha bulletin, STT 0.02% to 0.05% on futures from 1 Apr 2026 (Budget 2026): https://zerodha.com/marketintel/bulletin/445377/revision-in-stt-securities-transaction-tax-from-1st-april-2026
4. 5paisa, STT hike effective 1 Apr 2026: https://www.5paisa.com/news/stt-hike-on-fo-to-take-effect-from-april-1-amid-rising-options-activity
5. NSE press release, March 2023 (Jan 2021 surcharge of 6% withdrawn; IPFT Rs 10 per crore from 1 Apr 2023): https://nsearchives.nseindia.com/web/sites/default/files/inline-files/PR%20NSE%20rolls%20back%20increase%20in%20transaction%20charges.pdf
6. Zerodha bulletin, 15 Mar 2024 (futures 0.00190% to 0.00188% from 1 Apr 2024): https://zerodha.com/marketintel/bulletin/373933/revision-in-nse-transaction-charges-from-april-1-2024
7. NSE circular FA73061 (Rs 173 + Rs 10 IPFT per crore from 1 Oct 2024): https://nsearchives.nseindia.com/content/circulars/FA73061.pdf ; Zerodha bulletin, 28 Sep 2024: https://zerodha.com/marketintel/bulletin/391488/revision-in-transactions-charges-from-1st-october-2024
8. NSE circular of 27 Feb 2026, summary (IPFT cut, total unchanged, from 1 Mar 2026): https://teamleaseregtech.com/updates/article/53156/nse-issued-a-circular-regarding-the-revision-in-transaction-charges
9. Zerodha charges page (brokerage, SEBI fee Rs 10 per crore, GST 18%): https://zerodha.com/charges ; Motilal Oswal on the SEBI fee: https://www.motilaloswal.com/learning-centre/2023/6/what-is-the-sebi-turnover-charges
10. Zerodha bulletin, 1 Jul 2020 (stamp duty on futures 0.002%, buy side): https://zerodha.com/marketintel/bulletin/259741/uniform-stamp-duty-applicable-from-july-1-2020
11. Zerodha Z-Connect on SEBI circular SEBI/HO/MRD/TPD-1/P/CIR/2024/132 (minimum contract value Rs 15 lakh from 20 Nov 2024): https://zerodha.com/z-connect/business-updates/sebis-new-rules-for-index-derivatives-heres-whats-changing
12. No authoritative spread source found. Illustrative only, an impact cost of about 0.0023% for Nifty futures in a worked example: https://www.5paisa.com/finschool/course/equity-derivatives-course/the-nifty-futures/
13. Bailey, D. H. and Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality", Journal of Portfolio Management 40(5), 94-107, doi:10.3905/jpm.2014.40.5.094. Confirmed at https://papers.ssrn.com/abstract=2460551 ; formula cross-checked with https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio
14. Bailey, D. H. and Lopez de Prado, M. (2012), "The Sharpe Ratio Efficient Frontier", Journal of Risk 15(2), 3-44: https://www.risk.net/journal-risk/2223785/sharpe-ratio-efficient-frontier
15. Cited from memory, not looked up: Politis, D. N. and Romano, J. P. (1994), "The Stationary Bootstrap", JASA 89(428); Newey, W. K. and West, K. D. (1987), Econometrica 55(3).
