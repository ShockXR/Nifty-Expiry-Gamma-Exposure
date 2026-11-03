# Replicating Chhimwal, Pandey and Pandey (2025) on real NSE data

Paper: B. Chhimwal, V. Pandey and P. Pandey (2025), "Effect of multiple index derivative expiry on volatility, volume, and connectedness: a tale of two stock indices in India", *Review of Derivatives Research* 28(3), article 14.

**Every number in this note is REAL DATA** (third-party 1-minute Nifty 50 and Bank Nifty bars, validated against NSE's official daily index values, plus NSE's official F&O bhavcopy), unless it is marked **paper (printed)**. Printed values are typed in by hand from the paper's tables, only so that we can compare. A separate AI agent re-keyed all 223 of them blind against the PDF: 222 matched first time and the one difference is the paper's own typesetting slip, so no value changed (`docs/REKEY_CHECK.md`). Every one of our numbers comes from a file written by `experiments/replicate_paper.py` into `results/replication_nse/` (the file is named next to each table) or into `results/replication_nse_stdout.txt`.

## Findings

1. **The "after" sample (March to May 2024) replicates very closely.** With 360 minutes a day (09:15 to 15:14) and 1-minute log returns that carry the overnight gap in the first minute, we get exactly the paper's 21,239 Nifty returns. Our minimum, maximum and standard deviation match Table 1 to the printed digits. The mcsGARCH fit for Nifty gives alpha 0.0498, beta 0.9407 and log-likelihood 145,058 (paper (printed): 0.049363, 0.941048 and 145,063).
2. **The "before" sample, as the paper describes it (November 2023 to February 2024), does not replicate.** It has 82 trading days, not 64, and its return extremes, kurtosis and mcsGARCH estimates are far from Tables 1 and 2.
3. **A 64-day window from 14 November 2022 to 10 February 2023 matches the paper's "before" numbers.** We found it by scoring every 64-day window from July 2022 to February 2024 against Table 1. The paper's two "before" maxima equal returns in our data at 30 January 2023, 09:20 (Nifty) and 3 February 2023, 09:15 (Bank Nifty), to every printed decimal. The two "before" minima do not match exactly: ours are 0.5 and 1.2 basis points less negative, both in the 09:15 bar of 30 January 2023 (Section 4 traces this to one bar of data). The window's weekday counts also differ from the printed ones (Section 3). Numbers not used in the search agree: the ADF and Anderson-Darling statistics, the Table 2 volatility parameters (within ordinary sampling noise), and the volume changes in Table 5. **So our main question for Prof. Pandey is whether the "before" sample was November 2022 to February 2023.** If it was, both indices expired on Thursdays (weekly and monthly) in the "before" period. That fits Table 3's description of the two indices having "the same expiry structure".

   *Update (3 November 2026, P108):* a second search, registered in advance and run blind to this one ([P108_BLIND_WINDOW.md](P108_BLIND_WINDOW.md)), confirms the year, the 64-day length, the return definition and both maxima, but also scores the printed means and prefers **22 November 2022 to 20 February 2023**. Over 14 November to 10 February the Bank Nifty mean is -0.00000055, which prints as -0.000001 rather than the printed -0.000002; the higher moments used here favour starts of 14 to 17 November. The fair statement is a window starting between 14 and 23 November 2022 and ending between 10 and 21 February 2023 (the two candidates share 58 of 64 days). Results below use 14 November to 10 February.
4. **Intraday volatility shocks die out much faster after the change, but the size of the change is uncertain.** On the inferred window, the half-life of a shock falls from 359 to 72 minutes for Nifty (paper (printed) parameters imply 339 to 72) and from 162 to 38 minutes for Bank Nifty (printed: 185 to 40). The "before" half-lives are imprecise (Nifty 95% interval 197 to 1,255 minutes). The fall in persistence is clear for Nifty in every test. For Bank Nifty it is clear in a likelihood-ratio test, but its bootstrap interval just includes zero.
5. **Connectedness: the overall fall replicates, the hour-by-hour numbers do not.** On all expiry days, total connectedness falls from 87.9 (before) to 74.9 (Nifty expiry days) and 68.9 (Bank Nifty expiry days) on the paper's scale (paper (printed): 84.00 to 76.38 and 73.02). Both falls are clear in our bootstrap. The first-hour and last-hour totals of 99.42 and 105.22 in Table 3 are outside everything our 40-configuration grid produces. Each hour sub-sample has only three or four hours of data, and the results depend strongly on choices the paper does not report.
6. **Volume (Table 5) cannot be replicated minute by minute**, because no free minute volume exists for this period. NSE's official daily constituent volume for each index gives changes close to the paper's: Nifty +41% overall and +68% on Thursdays (printed medians imply +33% and +65%), Bank Nifty -11% (printed: -13%). The Nifty results survive day-level tests. The Bank Nifty fall does not survive them.
7. **Volatility (Table 6): the Nifty findings survive robust tests, most others do not.** Nifty volatility rose overall and on Thursdays, and both survive day-level tests. Thursday also stands out from the rest of its own week more than before. The paper's Bank Nifty finding (higher volatility on Wednesday, its new expiry day) is +16% in our data, but day-level tests do not find it significant and the week-relative test finds nothing special about Wednesday. The paper's minute-level test gives p-values as small as 10^-256 in our data, which shows why minute-level tests should not be trusted here.
8. **A few printed numbers look like typesetting slips** (Section 8). One example: Table 4's first-hour Nifty rows sum to 101. A blind second re-key confirmed each of them as printed in the PDF, so they are in the paper, not slips of ours (`docs/REKEY_CHECK.md`).
9. **Two small bugs in `src/niftygex/volatility.py`** were found, worked around here, and then fixed in the package (Section 10).

## 1. What we did, in plain words

The paper asks what happened when NSE moved Bank Nifty's expiry from Thursday to Wednesday while Nifty 50 stayed on Thursday. It compares a "before" and an "after" sample of 1-minute index data in four ways:

* **Table 1** describes the 1-minute returns.
* **Table 2** fits a volatility model (mcsGARCH) that splits each minute's variance into a daily level, a time-of-day pattern and a short-lived intraday part.
* **Tables 3 and 4** measure "connectedness": how much of each index's future volatility is explained by shocks to the other index (Diebold and Yilmaz 2012).
* **Tables 5 and 6** test whether per-minute volume and per-minute volatility changed (Mann-Whitney tests).

We rebuilt each step with this repository's tested code, ran it on real data, and put our numbers next to the printed ones. Where the paper does not say which setting it used (the VAR lag, the forecast horizon, the input series for connectedness, and which 360 minutes of the 375-minute session), we report a grid of reasonable settings rather than one pick. We also added tests that allow for the fact that minutes within a day are not independent.

### Words used below

* **Log return:** the change in the log of the index from one minute to the next, roughly the percentage change.
* **Half-life:** how many minutes it takes for an intraday volatility shock to lose half its effect. It equals ln(0.5) / ln(alpha + beta), where alpha + beta is the "persistence" in Table 2.
* **Connectedness total (paper scale):** the sum of the two cross shares in a 2 x 2 table. The standard Diebold-Yilmaz index is half of this number.
* **p-value:** roughly, how surprising the data would be if nothing had changed. Small means "probably a real change", but only if the test's assumptions hold.
* **Minute-level test:** treats every minute as an independent observation. **Day-level test:** first reduces each day to one number, then compares days. Minutes within a day share the same news and the same volatility level, so the day-level test is the honest one.
* **HAC:** a regression standard error that allows for correlation between nearby minutes (here, up to one day apart).
* **Bootstrap:** recompute the statistic on many resampled data sets to see how much it moves. We resample whole days (or blocks of days), never single minutes.
* **Week-relative comparison:** the expiry day's value divided by the average of the other days of the same week, on a log scale. It removes anything that affected the whole week, such as a volatile month.

## 2. Data and its validation

| Source | What | Used for |
| --- | --- | --- |
| `data/raw/nifty_1min.csv`, `banknifty_1min.csv` | third-party 1-minute open, high, low and close from community GitHub datasets (provenance in `minute_bars_source.json`); no volume | Tables 1, 2, 3, 4, 6 |
| `data/raw/index_close/`, `data/raw/extra_2022/index_close/` | NSE's official daily index files: open, high, low, close, constituent volume and turnover | validation, Table 5 |
| `data/raw/bhavcopy/`, `data/raw/extra_2022/bhavcopy/` | NSE's official F&O bhavcopy for NIFTY and BANKNIFTY | expiry calendar, Table 5 |

The November and December 2022 official files were downloaded for this note into `data/raw/extra_2022/` (gitignored) with the repository's own `scripts/download_nse.py`. Official dates are taken from the file names, because some April 2023 NSE files write the date month-first inside the file.

**Validation** (`validation.csv`, `validation_by_day.csv`). From 20 October 2022 to 30 June 2024 we compared each day's first minute open, highest high, lowest low and last close with NSE's official values (414 days per index):

| | Nifty 50 | Bank Nifty |
| --- | --- | --- |
| Open: median and largest difference | 0 and 7.9 basis points | 0 and 3.1 basis points |
| High: median and largest difference | 0.22 and 10.3 bp | 0 and 31.4 bp |
| Low: median and largest difference | 0.27 and 6.8 bp | 0 and 8.8 bp |
| Close: median and largest difference | 4.1 and 51.2 bp | 5.4 and 59.2 bp |
| Days with open, high or low off by more than 5 bp | 4 | 14 |

A basis point (bp) is 0.01%. Opens match exactly on almost every day. On the few flagged days the minute bars miss an extreme tick that NSE's official high or low contains, so the official range is slightly wider. In `validation_by_day.csv` the minute bars never go above the official high or below the official low, apart from one Nifty day by 0.5 bp. Closes differ by a few basis points because NSE's official index close is built from the constituents' closing prices (an average over the last 30 minutes), not from the last trade. Inside the three samples the flagged days are 29 December 2022 (Nifty) and five Bank Nifty days in the inferred window (largest 31.4 bp on 29 December 2022), and two Nifty opens in the after window (13 March and 15 April 2024). Nothing was flagged in the stated window.

Other data notes: the Nifty minute file lacks the 24 October 2022 evening Muhurat session; Bank Nifty has 369 of 375 bars on 23 April 2024, so we have 21,233 Bank Nifty returns after the change, not 21,239. Weekend special sessions in the minute files (12 November 2023, 20 January 2024, 2 March 2024 and 18 May 2024) are left out, as the paper's counts imply.

**Expiry calendar** (`expiry_calendar.csv`). An expiry day is a day on which a contract trades for the last time on its stated expiry date, read from the bhavcopy. We do not assume weekdays. This picks up holiday moves: 25 January 2023 (Wednesday, because Thursday 26 January was a holiday), Nifty on Wednesday 10 April 2024, and Bank Nifty on Tuesdays 16 and 30 April 2024. A **monthly** expiry is a day on which an index futures contract expires. The simpler rule "last expiry of the month" would wrongly call Bank Nifty's weekly expiry of 31 January 2024 a monthly one, because it came after the January monthly expiry of 25 January. Bank Nifty's monthly expiry was on Thursday up to 29 February 2024 and on Wednesday from 27 March 2024.

## 3. Which sample, and which 360 minutes?

**Day counts** (`day_counts.csv`):

| Sample | Dates | Trading days | Wednesdays | Thursdays | Expiry days (Nifty, Bank Nifty) |
| --- | --- | --- | --- | --- | --- |
| before, stated in the paper | 1 Nov 2023 to 29 Feb 2024 | 82 | 18 | 18 | 18, 18 |
| before, inferred | 14 Nov 2022 to 10 Feb 2023 | 64 | 13 | 12 | 13, 13 |
| after | 1 Mar 2024 to 31 May 2024 | 59 | 11 | 12 | 13, 13 |
| paper (printed counts) | | 64 and 59 | 13 and 12 | 13 and 13 | |

The paper's counts (23,039 and 21,239 returns; 4,680 and 4,320 Wednesday minutes) are 64 and 59 days of 360 minutes, less one return. The after sample gives exactly 59 days. The stated before sample gives 82.

**The weekday counts do not match, even on the inferred window.** The inferred window has 13 Wednesdays and 12 Thursdays, where the paper's Tables 5 and 6 imply 13 and 13. The after window has 11 Wednesdays and 12 Thursdays, where the paper implies 12 and 13. Holidays explain our counts: Thursday 26 January 2023, Thursday 11 April 2024, and Wednesdays 17 April and 1 May 2024 had no trading. The printed Thursday counts (13 and 13) equal our Nifty expiry-day counts in both windows, which include the expiries moved to Wednesday 25 January 2023 and Wednesday 10 April 2024. So the paper's "Thursday" rows may be Nifty expiry days. We found no reading that gives 12 Wednesdays after the change (there are 11 Wednesdays and 13 Bank Nifty expiry days).

**Which 360 minutes** (`session_rule_inference.csv`). We scored each rule on how closely the standard deviation, minimum, maximum and kurtosis match Table 1 (0 is a perfect match). For the after sample:

| Session rule | Overnight gap in first minute | Nifty score | Bank Nifty score |
| --- | --- | --- | --- |
| 09:15 to 15:14 (drop last 15 minutes) | yes | 0.001 | 0.012 |
| 09:30 to 15:29 (drop first 15 minutes) | yes | 0.704 | 0.283 |
| 09:15 to 15:29 (all 375 minutes) | yes | 0.943 | 0.588 |
| any of the three | no | 1.81 to 1.96 | 1.81 to 1.96 |

So the paper most likely used the 360 bars from 09:15 to 15:14, with returns computed straight through from one day to the next. We use that as our main rule and report the other rules as a sensitivity check (`sensitivity_session_rule.csv`). The paper's kurtosis is excess kurtosis (normal = 0).

**Which before window** (`fingerprint_search.csv`). The best of 348 windows of 64 trading days is 14 November 2022 to 10 February 2023 (score 0.045). Its neighbours a few days earlier or later score almost as well. The best window starting on or after 1 November 2023 ranks 217th (score 6.21). Because this window was chosen to fit Table 1's standard deviation, minimum, maximum and kurtosis, those four numbers are not an independent check of it.

**How well do the checks that were not used in the search match?** One line per table. Each line gives the inferred window, then the printed value, then the stated window for contrast.

* **Table 1, other columns** (`table1_descriptive.csv`). ADF: -23.12 and -20.87 against printed -23.05 and -20.69, within 1% (stated window: -44.56 and -49.72). Anderson-Darling: 660.5 and 822.5 against 649.2 and 790.1, within 2% and 4% (stated: 1,004.8 and 1,133.2). Skewness: Nifty -0.57 against -0.52, but Bank Nifty -1.11 against -1.36 (stated: 3.19 and -2.68). Correlation: 0.853 against 0.854 (stated: 0.837).
* **Table 2** (`table2_mcsgarch.csv`, `table2_persistence.csv`). Alpha, beta and shape are close: Nifty 0.0246, 0.9735 and 8.99 against 0.0265, 0.9715 and 8.86; Bank Nifty 0.0321, 0.9636 and 7.33 against 0.0323, 0.9639 and 6.71. Both printed half-lives (339 and 185 minutes) lie inside our 95% intervals. The stated window gives alpha 0.0481 and 0.0586, about twice the printed values.
* **Table 5** (`table5_volume_daylevel.csv`, official constituent share volume). All six changes are within 2.4 to 8.7 percentage points of the changes implied by the printed medians, with the same sign (Nifty +41.3%, +23.5%, +67.7% against +32.6%, +19.3%, +65.0%; Bank Nifty -11.0%, +7.2%, -7.0% against -13.4%, +9.6%, -3.3%). The stated window gets three of the six signs wrong. One caveat: we picked share volume because its level fits the printed medians. Constituent turnover in rupees does not match (+64% and +76%).
* **Tables 3 and 4** (`tables34_grid_summary.csv`). A weak check. The all-expiry total is 87.9 against 84.00 (stated: 73.1), but across our grid of settings it ranges from 70.3 to 91.8 on the inferred window and 55.0 to 83.5 on the stated one, so it barely separates the two windows.
* **Table 6** (`table6_volatility.csv`). Not a useful check. Our medians are 25% to 30% below the printed ones in both windows, and the printed medians have only two significant digits.

These checks are not fully independent of the search. Skewness, Anderson-Darling and the GARCH fit respond to the same few large returns that drive the kurtosis and the extremes. Table 5 uses a different data set (NSE's daily file), so it is the most independent of them.

## 4. Tables, ours next to the paper

In every table below, "inferred" is the 14 November 2022 to 10 February 2023 window and "stated" is November 2023 to February 2024.

### Table 1: descriptive statistics of 1-minute returns (`table1_descriptive.csv`)

| Statistic | Nifty before (inferred) | Nifty before, paper (printed) | Nifty after | Nifty after, paper (printed) | Bank Nifty before (inferred) | Bank Nifty before, paper (printed) | Bank Nifty after | Bank Nifty after, paper (printed) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N | 23,039 | 23,039 | 21,239 | 21,239 | 23,039 | 23,039 | 21,233 | 21,239 |
| SD | 0.000358 | 0.000361 | 0.000355 | 0.000355 | 0.000519 | 0.000522 | 0.000445 | 0.000443 |
| Minimum | -0.008534 | -0.008585 | -0.007086 | -0.007086 | -0.017135 | -0.017257 | -0.008699 | -0.008699 |
| Maximum | 0.006221 | 0.006221 | 0.009408 | 0.009408 | 0.012023 | 0.012023 | 0.009905 | 0.009905 |
| Skewness | -0.57 | -0.52 | 0.53 | 0.53 | -1.11 | -1.36 | 0.63 | 0.64 |
| Excess kurtosis | 63.9 | 64.7 | 75.9 | 75.9 | 124.8 | 125.1 | 56.9 | 57.3 |
| Anderson-Darling | 660.5 | 649.2 | 493.0 | 492.4 | 822.5 | 790.1 | 469.5 | 466.2 |
| ADF | -23.12 | -23.05 | -29.62 | -29.63 | -20.87 | -20.69 | -145.90 | -145.61 |

Correlation between the two indices: 0.853 before (inferred) and 0.818 after (paper (printed): 0.854 and 0.818). The stated window gives very different numbers: for Nifty, N 29,519, minimum -0.0115, maximum 0.0135 and excess kurtosis 195; correlation 0.837.

**The maxima match exactly; the before minima do not.** On the inferred window both maxima equal the printed values to every printed decimal (Nifty 0.006221108, Bank Nifty 0.012023448). In the after window all four extremes match to the printed digits. The two before minima differ: ours are -0.008534 and -0.017135, the printed values are -0.008585 and -0.017257. Ours are 0.5 and 1.2 basis points less negative. Both are the 09:15 bar of Monday 30 January 2023, the bar that carries the overnight gap from Friday 27 January.

**Is the overnight gap measured from NSE's official close?** We tested this (`table1_overnight_variants.csv`, `table1_implied_reference_price.csv`). The answer is no. Measuring the gap from the official close moves the minima the wrong way (-0.008122 and -0.016757) and breaks all four after-window extremes, which match with our main rule. The close of the 15:29 bar does no better. We also worked out, for each extreme that sits in a first minute, which previous-day price would reproduce the printed value exactly. In five of the seven cases it is our 15:14 close, to within the printed precision. In the other two, the 30 January minima, it is 17,612.50 (Nifty) and 40,365.45 (Bank Nifty). Those are exactly the opening prices of the 15:15 bar on 27 January in our data; our 15:14 closes are 17,611.60 and 40,360.55. Using the 15:15 open on every day reproduces both before minima but breaks the Bank Nifty before maximum and every after-window extreme. Our reading: the paper's minute data had a slightly different 15:14 close on 27 January 2023 (a one-bar difference between data feeds), not a different way of building returns.

### Table 2: ARMA(1,1)-mcsGARCH(1,1) with Student-t errors (`table2_mcsgarch.csv`)

| Parameter | Nifty before (inferred) | Nifty before, paper (printed) | Nifty after | Nifty after, paper (printed) | Bank Nifty before (inferred) | Bank Nifty before, paper (printed) | Bank Nifty after | Bank Nifty after, paper (printed) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AR(1) | 0.538 | -0.055 | -0.398 | -0.628 | -0.551 | 0.619 | 0.230 | 0.750 |
| MA(1) | -0.514 | 0.078 | 0.419 | 0.639 | 0.534 | -0.642 | -0.232 | -0.793 |
| AR + MA | 0.025 | 0.023 | 0.021 | 0.011 | -0.017 | -0.023 | -0.002 | -0.042 |
| omega | 0.00212 | 0.00232 | 0.01029 | 0.01035 | 0.00459 | 0.00426 | 0.01950 | 0.01869 |
| alpha | 0.0246 | 0.0265 | 0.0498 | 0.0494 | 0.0321 | 0.0323 | 0.0538 | 0.0532 |
| beta | 0.9735 | 0.9715 | 0.9407 | 0.9410 | 0.9636 | 0.9639 | 0.9281 | 0.9297 |
| shape (t degrees of freedom) | 8.99 | 8.86 | 7.01 | 6.98 | 7.33 | 6.71 | 5.67 | 5.48 |
| half-life, minutes | 358.6 | 339.1 | 72.4 | 71.9 | 162.0 | 185.1 | 38.0 | 40.1 |
| log-likelihood | 158,900 | 158,616 | 145,058 | 145,063 | 151,238 | 148,500 | 139,620 | 139,758.5 |

The printed half-lives are computed by us from the printed alpha and beta. Our daily variance comes from an AR(1)-EGARCH(1,1)-t fitted on daily returns since 2017, in-sample for each period. Our ARMA terms are estimated first and the GARCH terms second, while the paper probably estimated them jointly (for example with rugarch). On the stated window the fit is quite different (Nifty alpha 0.0481, beta 0.9501, shape 6.37; Bank Nifty alpha 0.0586, half-life 68.9 minutes).

The AR and MA estimates swing widely (even their signs differ from the paper's) while AR + MA stays close to zero in every column. This is what ARMA(1,1) does on returns that are close to unpredictable: the two terms cancel. With robust standard errors, none of our AR or MA terms is significant (largest |z| = 1.51 on the inferred and after samples).

**Uncertainty in the half-life** (`table2_persistence.csv`, `table2_persistence_change.csv`):

| Sample | Half-life (min) | Delta-method SE | Profile-likelihood 95% interval | Day-block bootstrap 95% interval |
| --- | --- | --- | --- | --- |
| Nifty before (inferred) | 358.6 | 139.7 | 197 to 1,255 | 103 to 701 |
| Nifty after | 72.4 | 14.9 | 51 to 115 | 37 to 139 |
| Bank Nifty before (inferred) | 162.0 | 44.2 | 103 to 318 | 51 to 302 |
| Bank Nifty after | 38.0 | 8.3 | 25 to 60 | 11 to 102 |

The bootstrap draws 400 resamples of 5-day blocks and re-estimates the time-of-day pattern and the intraday GARCH each time. Did persistence fall? For Nifty, the bootstrap 95% interval for the change in alpha + beta is -0.0163 to -0.0006, and 99% of draws show a fall. A likelihood-ratio test of equal persistence gives p = 0.00006. For Bank Nifty the interval is -0.0591 to +0.0023 (94% of draws show a fall), and the likelihood-ratio p is 0.00002. The likelihood-ratio test trusts the model's minute-by-minute likelihood, so we give more weight to the bootstrap. The fall in persistence is in the same direction under all three session rules and with or without the overnight return (`sensitivity_session_rule.csv`). For example, the inferred Nifty half-life ranges from 264 to 361 minutes across these choices, and the after value from 68 to 73. Moving the 64-day window by a few days changes the before half-life from 318 to 370 minutes for Nifty and from 138 to 166 for Bank Nifty (`table2_window_sensitivity.csv`).

### Tables 3 and 4: connectedness (`tables34_connectedness.csv`, `tables34_changes.csv`, `tables34_grid_summary.csv`)

Main setting: VAR(2) on mcsGARCH conditional volatility, horizon 10 minutes, with VAR lags never crossing from one day (or one hour sample) to the next. Totals are on the paper's scale. Brackets give a 95% bootstrap interval from resampling whole days (or whole hours for the one-hour rows).

| Sub-sample | Before (inferred) | Before, paper (printed) | After, Nifty expiry days | After Nifty, paper (printed) | After, Bank Nifty expiry days | After Bank Nifty, paper (printed) |
| --- | --- | --- | --- | --- | --- | --- |
| All expiry days | 87.9 [81.1, 90.9] | 84.00 | 74.9 [70.1, 79.1] | 76.38 | 68.9 [64.8, 74.1] | 73.02 |
| Monthly expiry days | 81.1 [69.9, 88.6] | 74.70 | 67.7 [62.9, 79.6] | 74.58 | 74.1 [69.2, 76.9] | 72.74 |
| First hour, monthly expiry | 81.6 [70.0, 87.5] | 99.42 | 67.6 [50.6, 81.6] | 80.18 | 81.0 [78.4, 83.7] | 70.97 |
| Middle hour, monthly expiry | 78.8 [42.7, 86.1] | 68.64 | 50.2 [45.9, 85.9] | 68.81 | 58.4 [53.7, 66.5] | 66.54 |
| Last hour, monthly expiry | 84.7 [77.3, 89.6] | 105.22 | 77.4 [74.8, 92.3] | 66.44 | 80.3 [70.8, 87.7] | 74.43 |

On the inferred window, all expiry days give cross shares of 49.0 (Nifty from Bank Nifty) and 38.9 (Bank Nifty from Nifty). The printed values are 42.71 and 41.29. Over all days, connectedness is 85.4 [82.7, 87.7] before and 74.1 [71.9, 76.3] after. Each monthly-expiry row uses 3 days before and 3 days after, so each one-hour row uses only 180 minutes. The middle hour is 11:45 to 12:44 and the last hour 14:15 to 15:14, which are the middle and last hours of the 360-minute session.

**Change from before (inferred) to after**, with 95% bootstrap intervals:

| Comparison | Ours | Paper (printed difference) |
| --- | --- | --- |
| All days | -11.3 [-14.2, -7.7] | not reported |
| All expiry days, to Nifty expiry days | -13.1 [-19.1, -4.6] | -7.62 |
| All expiry days, to Bank Nifty expiry days | -19.0 [-24.4, -10.2] | -10.98 |
| Monthly, to Nifty monthly | -13.4 [-23.8, 2.8] | -0.12 |
| Monthly, to Bank Nifty monthly | -7.0 [-16.9, 3.5] | -1.96 |
| Last hour, to Nifty / Bank Nifty | -7.3 [-14.4, 6.2] / -4.4 [-15.0, 8.2] | -38.78 / -30.79 |

Every one-hour change has an interval that includes zero. On the stated window, no change is clearly different from zero apart from one (first hour, to Bank Nifty: +16.1 [9.9, 55.0], the opposite direction to the paper). Over all days the stated window gives 76.1 before and -2.0 [-5.8, 1.5] for the change.

**How much do the unreported settings matter?** Across 5 input series (conditional volatility, the intraday GARCH part only, returns, absolute returns, Rogers-Satchell volatility), VAR lags 1, 2, 5 and AIC-chosen, and horizons 10 and 30, totals can move a great deal. One example is 16.7 to 82.9 for the last hour of Bank Nifty's monthly expiries after the change. Using the inferred before window and the after window, 12 of the 15 printed totals fall inside our grid's range. The three that do not are the before first hour (our range 55.3 to 98.9 against 99.42), the before last hour (44.1 to 86.5 against 105.22), and the after Nifty first hour (54.3 to 73.7 against 80.18). On the stated before window, three of the five before totals fall outside: all expiry days (55.0 to 83.5 against 84.00), the first hour (47.7 to 78.6 against 99.42) and the last hour (55.0 to 82.4 against 105.22). A total above 100 needs both cross shares above 50%. The largest cross share anywhere in our grid for the before last hour is 51.5%. Leaving out the first minute of each day (the overnight gap) changes the before first-hour total only from 81.6 to 79.4.

**Direction of spillover** (net = "to others" minus "from others"). Over all days on the inferred window, Nifty is a net receiver (net -11.2). After the change it is close to balanced (net -1.2). The paper says Nifty is a net receiver and that this gets stronger after the change. We see the first part but not the second. `fig_rolling_connectedness.png` shows the rolling version (one-day window), in the spirit of the paper's Figure 1.

### Table 5: volume, replaced by day-level measures (`table5_volume_daylevel.csv`)

**What differs, plainly.** The paper tests per-minute volume. Free minute volume for an index does not exist for this period, so we did not invent one. The paper's per-minute medians (for example 478,000 for Nifty before) look like the traded volume of the index's constituent stocks. NSE's official daily file reports this as "Volume" for each index. Our median daily value divided by 360 is 623,922 for Nifty before and 881,297 after, which is the same order as the printed per-minute medians (478,000 and 634,000). A per-minute median is naturally below a per-minute average. So we test the official **daily** constituent volume, and also F&O activity from the bhavcopy (option premium turnover, notional turnover, quantity). Contract counts are not comparable across these samples because lot sizes changed (Nifty 50 to 25 during April 2024; Bank Nifty 25 to 15 in 2023).

Inferred window to after, official daily constituent volume:

| Index, days | Change in median, ours | Change implied by paper (printed) medians | Day-level Mann-Whitney p | HAC p | Bootstrap p | Week-relative change (p) |
| --- | --- | --- | --- | --- | --- | --- |
| Nifty, all days | +41.3% | +32.6% | < 0.0001 | 0.0001 | < 0.001 | not applicable |
| Nifty, Wednesdays | +23.5% | +19.3% | 0.028 | 0.051 | 0.023 | -0.07 (0.25) |
| Nifty, Thursdays | +67.7% | +65.0% | 0.0006 | < 0.0001 | < 0.001 | +0.20 (0.017) |
| Bank Nifty, all days | -11.0% | -13.4% | 0.046 | 0.11 | 0.18 | not applicable |
| Bank Nifty, Wednesdays | +7.2% | +9.6% | 0.86 | 0.77 | 0.74 | +0.14 (0.082) |
| Bank Nifty, Thursdays | -7.0% | -3.3% | 0.71 | 0.72 | 0.82 | +0.09 (0.54) |

The week-relative column is the change in log(day's value / average of the other days that week). For example, +0.20 means Thursday now stands out about 22% more from its own week than before. On the stated window the same measure gives Nifty +10.9%, Wednesday -3.8% and Thursday +15.9%, and Bank Nifty +3.2%, +22.1% and +5.8%, none of which resembles the paper.

F&O activity (inferred to after): Nifty option premium turnover rose 32.3% overall and 179.5% on Thursdays, where it stands out sharply from its own week (week-relative +0.74, p < 0.0001). Bank Nifty option premium fell 34.4% on Thursdays and rose 35.3% on Wednesdays. Wednesday now stands out from its week (+0.40, p = 0.0006), which is what moving the expiry should do.

### Table 6: per-minute Rogers-Satchell volatility (`table6_volatility.csv`)

We use the square root of the Rogers-Satchell variance of each 1-minute bar.

| Index, days | Our median before (inferred) / after | Paper (printed) medians | Change in median | Minute-level p (paper's test) | Day-level p | HAC p | Bootstrap p | Week-relative change (p) | Paper (printed) p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nifty, all | 0.000166 / 0.000198 | 0.00024 / 0.00026 | +19.7% | 3e-256 | 0.0003 | 0.0003 | < 0.001 | not applicable | 0.000 |
| Nifty, Wednesday | 0.000156 / 0.000189 | 0.00023 / 0.00023 | +21.1% | 7e-53 | 0.18 | 0.12 | 0.083 | +0.05 (0.86) | 0.061 |
| Nifty, Thursday | 0.000164 / 0.000245 | 0.00024 / 0.00027 | +49.3% | 1e-199 | 0.010 | 0.00005 | 0.002 | +0.22 (0.005) | 0.000 |
| Bank Nifty, all | 0.000255 / 0.000292 | 0.00034 / 0.00034 | +14.8% | 6e-149 | 0.0017 | 0.0083 | 0.0005 | not applicable | 0.087 |
| Bank Nifty, Wednesday | 0.000251 / 0.000292 | 0.00033 / 0.00036 | +16.4% | 4e-45 | 0.15 | 0.13 | 0.089 | +0.06 (0.30) | 0.000 |
| Bank Nifty, Thursday | 0.000257 / 0.000322 | 0.00034 / 0.00034 | +25.5% | 4e-65 | 0.069 | 0.047 | 0.017 | +0.06 (0.47) | 0.079 |

On Bank Nifty's own expiry days (Wednesday or the holiday-shifted day) its volatility rose 11.7%, with a day-level p of 0.36. Its expiry day does not stand out from its own week more than before (week-relative -0.02, p = 0.88). Our medians are 25% to 30% below the printed ones in every column, so the paper's volatility measure probably differs from ours in some detail. We tried a few variants (Parkinson, Garman-Klass, previous close as open, rolling averages), and none matched all four printed medians. This does not change any Mann-Whitney p-value, because ranks do not change when one measure is rescaled. It does matter for comparing levels. The percentage changes barely move with the session rule (for example, Nifty on Thursday +49.3% to +51.2%; `sensitivity_session_rule.csv`).

## 5. What replicated

* The after sample, almost number for number: Table 1, the Table 2 volatility parameters and log-likelihood, and Table 1's ADF and Anderson-Darling statistics.
* The before sample, **if** it is November 2022 to February 2023: Table 1 closely (both maxima exactly; the minima differ because of one bar of data), Table 2 within ordinary sampling noise, and the size of Table 5's volume changes. The weekday counts do not match (Section 3).
* The big picture of the paper: Nifty volume and volatility rose, especially on Thursdays. Connectedness between the two indices fell, most clearly on expiry days. Intraday volatility shocks fade faster after the change.

## 6. What did not replicate, and the most likely reasons

| Result | What we see | Most likely reason |
| --- | --- | --- |
| The before sample as described | 82 days, different extremes and parameters | The sample may have been November 2022 to February 2023 (Section 3) |
| Before minima on the inferred window | -0.008534 and -0.017135 against printed -0.008585 and -0.017257, both at 30 January 2023, 09:15 | A different 15:14 close on 27 January 2023 in the paper's data; NSE's official close does not explain it (Section 4) |
| Weekday counts | inferred window 13 Wednesdays and 12 Thursdays (printed 13 and 13); after window 11 and 12 (printed 12 and 13) | Holidays on two Thursdays and two Wednesdays; the printed Thursday counts equal our Nifty expiry-day counts, so those rows may be expiry days; we cannot explain 12 Wednesdays |
| First and last hour connectedness of 99.42 and 105.22 | 81.6 and 84.7; no setting in our grid exceeds 98.9 | Unreported VAR settings or input series, and only three or four hours of data per row |
| One-hour changes after the policy | intervals include zero | Too little data per row (180 to 240 minutes) |
| AR(1) and MA(1) values and their stars | different values and signs, none significant | The two terms nearly cancel and are weakly identified |
| Table 6 levels | our medians 25% to 30% lower | A different volatility definition or scaling |
| Bank Nifty: no overall volatility change, but a rise on Wednesday | the reverse: a clear overall rise; the Wednesday rise is not significant at day level | Minute-level tests; and in our data Bank Nifty volatility rose on all days |
| Bank Nifty volume fall | -11% matches the size, but day-level tests are not convincing | 64 and 59 days are a small sample for daily volume |

## 7. What the robust tests change

* **Minute-level tests are far too confident.** Every minute-level Mann-Whitney p-value in Table 6 is below 10^-13, even where day-level tests find nothing. Repeated minutes from the same day are not independent evidence. Experiment E2 in this repository shows by simulation how much this inflates false positives.
* **What survives:** the rise in Nifty volume and volatility overall and on Thursdays, the jump in Nifty option premium on its expiry day, the fall in connectedness on expiry days, and the fall in Nifty's intraday persistence.
* **What does not clearly survive:** the Wednesday effects for both indices, Bank Nifty's volume fall, every one-hour connectedness change, and (just barely) the fall in Bank Nifty's persistence in the bootstrap.
* **The week-relative view adds something.** Nifty's Thursday now stands out from its own week in volume, option premium and volatility. Bank Nifty's Wednesday now stands out in option premium, but not in index volume or volatility.

## 8. Questions to discuss with Prof. Pandey

These are offered as careful questions. Each may have a simple answer that the paper's space did not allow for.

1. Was the before sample November 2022 to February 2023, rather than November 2023 to February 2024? Our evidence: the 64-day count, two maxima that match to every printed decimal, Table 2's parameters, Table 5's volume changes, and Table 3's description of a "same expiry structure", which is only true before September 2023. If so, the before-and-after comparison covers both the weekly move (September 2023) and the monthly move (March 2024). It also includes a year of other market changes.
2. Which 360 minutes were used? Our evidence points to 09:15 to 15:14, with returns running across days (the overnight gap sits in the first minute). If so, the "last trading hour" ends at 15:14 and leaves out the final 15 minutes, which are half of the 30-minute window (15:00 to 15:30) over which NSE computed the official closing prices, and so the index close used to settle index options, until its closing auction began on 3 August 2026.
3. For connectedness: was the VAR run on mcsGARCH conditional volatility or on returns? Which lag and horizon were used? Were the expiry days joined end to end, so that VAR lags run from one day into the next? Our grid shows these choices can move totals by more than the before-to-after change itself.
4. How was a total above 100 (105.22) obtained? It requires both cross shares to exceed 50%, which none of our settings produces for that sub-sample.
5. Which volume series is in Table 5? Constituent share volume fits the printed levels and changes. Knowing the source would let us match it exactly.
6. Was the volatility in Table 6 the square root of the per-minute Rogers-Satchell variance, or a scaled or averaged version?
7. Would the authors consider day-level tests (or HAC or a day-block bootstrap) alongside the minute-level Mann-Whitney test? In our data they change the conclusions for the Wednesday results.
8. Small printed-number points (`paper_printed_checks.csv`), possibly typesetting slips. A blind second re-key confirmed every value below as printed in the PDF (`docs/REKEY_CHECK.md`, with page references), so none of them is a transcription slip of ours:
   * Table 1, Bank Nifty after: the Anderson-Darling statistic is printed as 466,153, with a comma where the rest of the row has a decimal point. We read it as 466.153 (ours: 469.5).
   * For Bank Nifty before, the printed log-likelihood and AIC imply N = 22,680 (63 days of 360 minutes), not 23,039.
   * Table 4, Panel I, first hour: both rows sum to 101.00, not 100.
   * Table 4, Panel II, first hour: the cross shares sum to 71.97, but the total is printed as 70.97.
   * Table 5, Nifty on Thursday: the printed medians imply +65%, and the text says 61%.
   * Tables 5 and 6 statistics: read as rank sums, all six Table 5 statistics are possible and reproduce their printed p-values (including 0.0011 for Bank Nifty on Thursday). In Table 6, five of six cannot be reconciled with their p-values under either reading (three are larger than the largest possible value).
9. Which days are in the Wednesday and Thursday rows of Tables 5 and 6? The printed minute counts imply 13 and 13 days before and 12 and 13 after. Our calendar has 13 and 12 before and 11 and 12 after; the Thursday counts would match if those rows were Nifty expiry days.
10. Which minute-data source was used? It seems to agree with ours on every extreme except one bar, the 15:14 close of 27 January 2023.

## 9. How to reproduce

```
OMP_NUM_THREADS=1 python experiments/replicate_paper.py            # about 3 minutes on 4 cores
OMP_NUM_THREADS=1 python experiments/replicate_paper.py --trim 375_full   # another session rule as the main one
OMP_NUM_THREADS=1 python -m pytest -q tests/test_replication.py
python experiments/rekey_check.py                                  # blind second transcription of the printed values (docs/REKEY_CHECK.md)
```

The script reads only `data/raw/` and writes `results/replication_nse/*.csv`, two figures (`fig_diurnal.png`, `fig_rolling_connectedness.png`) and `results/replication_nse_stdout.txt`. All random draws are seeded. New helper code is in `src/niftygex/replication.py`.

## 10. Notes on the repository's code

* **Bug: `McsGarch.fit` with an empty minute column** (`src/niftygex/volatility.py`, lines 326 and 333). If one minute of the day has no valid return on any day (for example `R[:, 0] = np.nan` to drop the overnight gap), its time-of-day factor becomes 0, the normalised residual becomes 0/0 = NaN, and the GARCH likelihood is never finite. The fit then returns its starting values (alpha 0.03, beta 0.95, nu 8) with `converged=False` and log-likelihood -1e300, with only a RuntimeWarning. The replication script works around this by dropping such columns before fitting. **Fixed** after this note was written: such minutes now get the average time-of-day factor (`McsGarch._pattern`), with a regression test (`tests/test_volatility.py`). The replication results do not change, because the script already dropped those columns, and neither do the forecasting results in `results/pipeline_nse/`, which never had an empty minute.
* **Bug: `McsGarch.diurnal()` with no argument** (lines 298 to 305) raises `TypeError: object of type 'NoneType' has no len()` at line 301, although `expiry_mask=None` is the default. `fit` and `filter` are not affected, because they always pass a mask. **Fixed:** with no mask it now returns the ordinary-day pattern as a single row.
* **Trap (not a bug):** `day_level_mwu`, `hac_mean_difference` and `day_cluster_bootstrap` in `inference.py` expect a (days x minutes) array. A one-dimensional vector of daily values is silently treated as a single day. Pass `values.reshape(-1, 1)`, and give `hac_mean_difference` an explicit `maxlags`.
* **Bootstrap for connectedness:** resampling 30-minute blocks biased the total downward (the point estimate sometimes fell outside its own interval), because conditional volatility is far more persistent than 30 minutes. We resample whole days or whole hours instead (`replication.segment_bootstrap_connectedness`).
