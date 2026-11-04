# The anchor paper, read closely

Chhimwal, B., Pandey, V. and Pandey, P. (2025). Effect of multiple index derivative expiry on volatility, volume, and connectedness: a tale of two stock indices in India. *Review of Derivatives Research*, 28(3), article 14. doi:10.1007/s11147-025-09221-8. Authors' affiliations: FORE School of Management (Chhimwal, V. Pandey) and IIT Bombay (P. Pandey).

The paper is paywalled, so this repository holds only our own summary and a handful of numeric values used as inputs to arithmetic checks. Read the paper itself for anything quoted here.

## 1. Motivation and question

NSE moved Bank Nifty's weekly expiry from Thursday to Wednesday from 6 September 2023, and its monthly and quarterly expiries to Wednesday from early 2024 (the paper says March 1, 2024 in one place and February in another). Nifty 50 stayed on Thursday. The stated aim of the exchange was to spread expiry activity across the week and reduce expiry-day volatility. The paper asks what actually happened to (a) volatility, (b) trading volume and (c) volatility connectedness between the two indices.

## 2. Data

* 1-minute Nifty 50 and Bank Nifty index data.
* Before: November 2023 to February 2024. After: March 2024 to May 2024.
* Observations: 23,039 before and 21,239 after (Table 1). Both are exact multiples of 360 to within one (64.0 and 59.0 days of 360 minutes), and the weekday subsamples (4,680 and 4,320) are exactly 13 and 12 days of 360 minutes. So the paper appears to use 360 minutes per day, not the full 375-minute session, over 64 and 59 trading days. The introduction mentions "90 days". The data vendor and the treatment of the first and last minutes are not reported.
* Note on the design: the "before" window already had Bank Nifty *weeklies* on Wednesday (they moved in September 2023). So the before/after split isolates the monthly and quarterly move, combined with whatever else changed between the two windows. The authors acknowledge this and list the isolated effect of the September 2023 weekly move as future work.

## 3. Methods

1. **mcsGARCH** (Engle and Sokalska 2012): ARMA(1,1) mean, multiplicative variance with a daily AR(1)-EGARCH(1,1) component, a diurnal component and an intraday GARCH(1,1) with Student-$t$ errors. Fitted separately before and after, per index. Parameters in Table 2.
2. **Diebold-Yilmaz (2012) connectedness** from a generalized variance decomposition. Reported for the whole sample (Figure 1) and for sub-samples (Tables 3 and 4): all expiry days, monthly expiry days, and the first, middle and last trading hour of monthly expiry days. The VAR lag, horizon, rolling window and the exact input series (returns or conditional volatilities) are not reported.
3. **Mann-Whitney-Wilcoxon tests** on per-minute volume (Table 5) and per-minute Rogers-Satchell volatility (Table 6), before versus after, overall and separately for Wednesdays and Thursdays.

## 4. Findings, as the paper states them

* Connectedness between Nifty 50 and Bank Nifty fell after the change (paper-scale total about 84 before, 73 to 76 after), and fell most around Bank Nifty's own expiry.
* Connectedness is highest in the first and especially the last trading hour of monthly expiry before the change (paper-scale 99.42 and 105.22).
* Nifty 50 median per-minute volume rose 32% overall, 19% on Wednesdays and 61% on Thursdays (its own expiry day). Bank Nifty volume fell 13% overall.
* Nifty 50 volatility rose overall and on Thursday; Bank Nifty volatility rose on Wednesday.
* Interpretation offered (not tested): traders roll from Wednesday's Bank Nifty expiry into Thursday's Nifty expiry; retail traders are drawn to cheap expiry-day premiums; expiry pressure is now concentrated within the index that is expiring.
* Policy reading: the change did not reduce expiry-day volatility, contrary to its aim.

## 5. What is and is not in the paper

| Item | In the paper? |
| --- | --- |
| Out-of-sample forecasts | No. All models are fitted in-sample per period. |
| Option positioning (OI, gamma, participant data) | No. The "why" is conjecture. |
| Standard errors or confidence intervals for connectedness | No. |
| Dependence-robust tests for volume and volatility | No. Wilcoxon on pooled minutes. |
| Code or data release | No ("No datasets were generated or analysed"). |
| Post-November-2024 regime (single weekly expiry) | No. Mentioned only in the conclusion via the January 2025 Bank Nifty change back to Thursday. |

## 6. Our checks of the published numbers

All from `experiments/paper_checks.py` (E5, E6). These are offered as careful-reader notes to discuss with the authors, not as criticisms of the conclusions.

1. **Connectedness scale.** Totals are the sum of off-diagonal shares, which is twice the standard DY index for two variables (84.00 is 42.00 on the DY scale). Comparisons within the paper are unaffected.
2. **Cross shares above 50%** in the last hour before the change (52.79 and 52.43) are possible only with strong lagged spillovers between the two series. Plausible, but it would help to know the VAR input and settings.
3. **ARMA(1,1) estimates cancel.** In all four Table 2 columns $\phi + \theta$ is between $-0.04$ and $+0.02$, and the implied first autocorrelation of returns is under 0.04 in absolute value. Large, opposite AR and MA terms are what ARMA(1,1) produces on near-white-noise data (weak identification); see E5 for the simulation.
4. **Significance stars.** For Nifty 50 before, the AR(1) and MA(1) estimates are marked ** but their printed z statistics are $-0.26$ and $0.37$. One of the two is a typo.
5. **Mann-Whitney statistics.** Three entries in Table 6 (Bank Nifty Wednesday, Nifty Thursday, Bank Nifty Thursday) exceed the largest value the statistic can take for those sample sizes (for example 207,455,847.5 against a maximum near 31 million). These look like misplaced digits. The p-values may still be right.
6. **Intraday persistence.** Table 2 implies the half-life of an intraday volatility shock fell from about 339 to 72 minutes for Nifty and from about 185 to 40 minutes for Bank Nifty. The paper does not discuss this, and it may be one of its more interesting results. Caution: near 1 the half-life is very sensitive to small errors in persistence. In our simulation at the paper's sample size (64 days, daily variance estimated by EGARCH; experiment E1a), estimated persistence was biased upward by about 0.002, which alone would roughly triple an implied half-life of about 230 minutes. So this needs standard errors before it is read as a finding.
7. **Wilcoxon on minutes.** The tests treat thousands of minutes from 12 or 13 days as independent. E2 measures how much that inflates false positives.

## 7. Limitations the paper itself names, and others

* Named by the authors: the September 2023 weekly change is not isolated.
* Short windows (64 and 59 days), with only about three or four monthly expiries on each side for the hourly sub-samples.
* No mechanism test.
* No regime after November 2024, when SEBI allowed one weekly benchmark expiry per exchange, and no regime after September 2025, when Nifty weekly expiry moved to Tuesday.

## 8. Extensions this prototype builds

1. **Mechanism.** Add dealer gamma exposure from the option chain and test whether it explains or forecasts expiry-day volatility (the forecasting contest and GEX ablation).
2. **Out of sample.** Turn the paper's in-sample mcsGARCH into a forecasting baseline and score it day by day.
3. **Stronger inference.** Day-level tests, HAC and block bootstrap for volume and volatility; bootstrap intervals for connectedness.
4. **New natural experiments, same design.** 20 November 2024 and 1 September 2025.
5. **Expiry-specific diurnal curve** inside mcsGARCH.
