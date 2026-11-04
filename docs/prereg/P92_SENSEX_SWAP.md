# P92 pre-registration: on its own expiry day, does the expiring index lead the volatility spillover between Nifty and Sensex?

Written 3 November 2026. **This document, the analysis script `experiments/sensex_swap.py` and its tests
`tests/test_sensex_swap.py` were committed to git before any connectedness, spillover, volatility or other outcome of
this study was computed on real data.** The git commit time stamp is the evidence; the phase 2 script prints the
commit of this file into `results/sensex_swap/run_info.json` (`prereg_commit`). Later changes are listed as deviations
in the write-up `docs/SENSEX_SWAP.md` (section 12). Everything below is about **REAL DATA** (NSE and BSE index minute
bars and exchange calendars); the only synthetic data are in the unit tests.

**What was computed on real data before this was committed (phase 1, `python experiments/sensex_swap.py --coverage`).**
Only the calendar and data coverage: which days have bars for both indices, how many minutes are missing on each day,
which index expires on each day, and how many days fall into each comparison cell (section 5). To decide how to treat
the closing auction, the share of minutes with an unchanged price between 15:16 and 15:28 was counted for both indices
(section 10), and, as a data-quality look, the share of unchanged one-minute closes from 09:16 to 15:14 pooled over
all days of each period (Nifty about 0.4%, Sensex below 0.05%). No return series, volatility, VAR or connectedness was
computed for Nifty and Sensex. The minimum detectable effects (section 6.4) use only the cell counts and a variance
taken from another study's published results file.

**What the author had already seen.** The Nifty and Bank Nifty connectedness study (`docs/CONNECTEDNESS_GAMMA.md`,
including its per-day file), the replication of the anchor paper (`docs/REPLICATION_NSE.md`) and the Sensex confound
study (`docs/SENSEX_CONFOUND.md`), which found that Nifty's end-of-day variance is higher on Sensex expiry days and that
the whole-day variance is not. None of these looked at connectedness or spillovers between Nifty and Sensex; the
project lists that pair as open (`alternatives/ALTERNATIVES.md`, A4).

## 0. In plain words (for a beginner)

* **Volatility spillover.** Nifty 50 (NSE) and Sensex (BSE) are two Indian stock indices that share most of their
  large companies, so they move almost together. Minute by minute, a burst of volatility in one is followed by a burst
  in the other. A **VAR** is a model that predicts each index's next-minute volatility from the last two minutes of
  *both* indices. From it, the **Diebold-Yilmaz** method asks: when we forecast Sensex volatility ten minutes ahead,
  what share of our error comes from surprises in Nifty ("Nifty to Sensex")? And the other way round ("Sensex to
  Nifty")?
* **NET.** NET = (Nifty to Sensex share) minus (Sensex to Nifty share), in percentage points. Positive NET means that
  on that day Nifty "led" more than it "followed": it was the **net transmitter**. We compute one NET per trading day.
* **Expiry day.** The day on which a weekly options contract ends. Traders close and hedge positions in the expiring
  index, which can make that index move first.
* **The natural experiment.** From January to August 2025, Sensex options expired on Tuesday and Nifty options on
  Thursday. On 1 September 2025 the exchanges swapped: Nifty moved to Tuesday and Sensex to Thursday. So the *same
  weekday* switched from being one index's expiry day to the other's. As far as we know, nothing else about Tuesdays
  or Thursdays changed on that date, apart from the other contracts of each exchange moving with its index (section 2).
* **The question (H1).** If the expiring index tends to lead, then Tuesday's NET should go *up* after the swap (Nifty
  now expires on Tuesday) and Thursday's NET should go *down* (Sensex now expires on Thursday). We compare the change
  on Tuesday with the change on Thursday. This **difference in differences** removes anything that is special about
  Tuesdays in general, and anything that changed for all days at once (for example a calmer market in 2026).
* **How we judge it.** A **permutation test**: we pretend, many times, that the swap happened in a random set of
  weeks instead of the real ones, and see how often a pretend swap produces a difference as large as the real one.
  If that happens less than 5 times in 100, we call H1 supported.
* **Checks.** The same comparison on Monday against Wednesday (where nobody expires, though the next day's expiry
  did swap; see section 8), and at two fake swap dates where nothing changed. If the fake dates "move" too, the design
  is not trustworthy.
* **How big an effect can we see?** About 4.9 points of the difference in differences, which means that a
  Nifty-expiry day and a Sensex-expiry day on the same weekday would need to differ in NET by about 2.5 points. That
  is under the assumption that NET varies from day to day about as much as it did for Nifty and Bank Nifty
  (section 6.4). If the real effect is smaller, the test will probably miss it.

## 1. Background

The anchor paper (Chhimwal, Pandey and Pandey 2025, "CPP") fits Diebold-Yilmaz connectedness to minute-level
mcsGARCH volatility of Nifty and Bank Nifty and shows the two became less connected after their expiry days were
separated. Our replication confirms the fall on expiry days (`docs/REPLICATION_NSE.md`, finding 5). Our own
connectedness study builds the same volatility input and day-level connectedness and found, as a post-hoc check, that
total connectedness on Nifty expiry days was 2.75 points lower than on other days (p 0.18; `docs/CONNECTEDNESS_GAMMA.md`,
E2). Nothing in the project measures the *direction* of spillovers on expiry days, and nothing uses the Nifty and Sensex
pair.

**Why the expiring index might lead.** On its expiry day, positions in the expiring contract are closed, rolled or
hedged, and option dealers' hedges in the underlying futures are largest and change fastest near expiry. That trading
hits the expiring index first; the other index follows through the shared stocks and through arbitrage. If so, the
expiring index's volatility shocks should explain more of the other index's forecast error than the other way round,
which raises its NET. This is a hypothesis, not a known fact; at the one-minute scale the lead of one of two
near-identical indices could also be small or reflect how the two index values are computed and published.

## 2. Question and hypotheses

**Question.** On its own expiry day, does the expiring index become the net transmitter of volatility spillovers
between Nifty and Sensex?

**H1 (primary, fixed by the project lead before this document was written).** With NET_d defined in section 4,

    delta = [NET(Tue, after) - NET(Tue, before)] - [NET(Thu, after) - NET(Thu, before)] > 0,

one-sided, where NET(cell) is the average of NET_d over the days of that cell, "before" is 1 January 2025 to
31 August 2025 and "after" is 1 September 2025 to 2 November 2026 (the last day with bars for both indices in the
pinned files).

**What delta measures.** Write NET_d = baseline + weekday effect + period effect + theta_N (if only Nifty expires)
minus theta_S (if only Sensex expires) + noise. Every Tuesday and Thursday in the window is the expiry day of exactly
one index (section 5), so delta = theta_N(Tue, after) + theta_S(Tue, before) + theta_S(Thu, after) +
theta_N(Thu, before): the sum of four expiry effects, each allowed to differ by weekday and period. Under H1 each term
is positive, so delta is positive even if the effects differ in size (for example if Sensex options grew after the
swap). With one common effect theta for both indices, delta = 4 theta. The design identifies this sum, not the separate
Nifty and Sensex effects. Weekday effects that are constant over time and period effects that are common to Tuesday
and Thursday cancel. The **identifying assumption** is that without the swap the Tuesday-minus-Thursday gap in NET
would have stayed the same (the placebos in section 8 check this).

**The treatment is "which exchange's contracts expire", not only the weekly index option.** The swap is reported to
have moved all NSE equity derivatives (so also the Nifty and Bank Nifty monthly contracts and single-stock futures and
options) to Tuesday and all BSE ones to Thursday (the web address of the Moneylife source in
`data/external/sensex_expiry_calendar.csv` reads "nse-equity-fo-contracts-to-expire-on-tuesdays-bse-on-thursdays";
the page itself and the circulars were not read). Monthly expiry days are
kept in the primary test and left out in a robustness check.

## 3. Data

| Input | What | Source and status |
| --- | --- | --- |
| `data/raw/nifty_1min.csv` | Nifty 50 one-minute bars | technovusin/nifty50-historical-data, commit 7f94ced8 (`data/raw/minute_bars_source.json`); validated against NSE daily high, low and open (`docs/REAL_DATA_RESULTS.md`, R0) |
| `data/raw/sensex_1min.csv` | Sensex one-minute bars, 2 Jan 2023 to 2 Nov 2026 | technovusin/sensex-historical-data, commit 22e57b89 (`data/raw/sensex_1min_source.json`); same provider as the Nifty bars; **not** validated against BSE official values (BSE servers blocked from this machine) |
| `results/pipeline_nse_v2/daily_market.csv` | Nifty expiry days (`is_expiry`), monthly expiry days (`is_monthly_expiry`, from index futures expiries), NSE expiry regime | Read from NSE F&O bhavcopies by the project pipeline |
| `data/external/sensex_expiry_calendar.csv` | Sensex weekly expiry weekday: Friday from 15 May 2023 to 3 Jan 2025, Tuesday from 1 Jan 2025 to 31 Aug 2025, Thursday from 1 Sep 2025; holiday rule: previous trading day | Reported (news and broker summaries of BSE and SEBI circulars, circulars not read); consistent with 71 expiry dates seen in third-party option prices and with the Sensex index's own end-of-day spike (`docs/SENSEX_CONFOUND.md`, section 3) |
| `results/connectedness_gamma/sample_days.csv` | Published per-day Nifty and Bank Nifty connectedness | Used only for the variance assumption of section 6.4 |

**Input files at phase 1** (first 16 hex digits of SHA-256, `results/sensex_swap/run_info_coverage.json`):
`nifty_1min.csv` 8ab19906a0a350b4, `sensex_1min.csv` 471a519e0300fbf0, `daily_market.csv` b9f57f0875240996,
`sensex_expiry_calendar.csv` 5d9329e40c8c75b2, `connectedness_gamma/sample_days.csv` f663478536e920a6. Phase 2 records
the same digits; a difference is reported as a deviation.

**Calendar checks (REAL DATA, `results/sensex_swap/calendar_check.csv`).** In the sample (21 Nov 2024 to 2 Nov 2026)
the Sensex rule gives 102 expiry days, the same 102 that the Sensex confound study used. All 29 individually reported
Sensex expiry dates inside the sample are among them. The 102 Nifty expiry days read from the bhavcopies agree with
the weekday rule of `src/niftygex/expiry_calendar.py` (Thursday to 31 Aug 2025, Tuesday from 1 Sep 2025, previous
trading day on holidays) on every day except 2 Nov 2026, where the rule wrongly moves the 3 Nov expiry back because
the data stop on 2 Nov; the bhavcopy flags are used. No day in the sample is an expiry day of both indices.

**Holiday-shifted expiry days in the sample** (`results/sensex_swap/calendar_days.csv`): Nifty on Wednesday 9 Apr 2025
and 30 Apr 2025, and on Monday 20 Oct 2025, 2 Mar 2026, 30 Mar 2026 (a monthly expiry), 13 Apr 2026 and 19 Oct 2026;
Sensex on Wednesday 1 Oct 2025, 24 Dec 2025, 14 Jan 2026, 25 Mar 2026 and 27 May 2026.

**Coverage (REAL DATA).** 480 trading days from 21 Nov 2024 to 2 Nov 2026 have bars for both indices. One day is left
out: the Diwali Muhurat session of Tuesday 21 Oct 2025 (60 bars). On all 480 days both indices have all 360 bars of the
session used (09:15 to 15:14): **zero missing minutes** for either index. Weekend special sessions are not used.

## 4. Exact variable definitions

All volatility steps reuse the code of `experiments/connectedness_gamma.py` (imported, not copied, and not changed).

1. **Session and days.** Bars 09:15 to 15:14 (the "360_drop_last15" rule of the replication and the connectedness
   study), weekdays only. A day is used if both indices have at least 300 of the 360 bars (all 480 have 360).
2. **Returns.** One-minute log returns of the bar closes, read continuously (the first minute of a day carries the
   overnight gap), on the common days from 2 Jan 2023; daily returns from the last bar of each day
   (`build_pair_grids`, the connectedness study's `build_grids` with the index names as an argument; a test checks
   that it gives identical output on Nifty and Bank Nifty).
3. **Minute volatility.** mcsGARCH (Engle and Sokalska 2012) exactly as `connectedness_gamma.fit_regime`: ARMA(1,1)
   mean, Student t errors, a diurnal factor s per minute of the day, an intraday GARCH(1,1) part q, and a daily variance
   h from an AR(1)-EGARCH(1,1)-t fitted on daily returns from 3 Jan 2023 to the last day of the fit group (the
   connectedness study used daily returns from 2017; the Sensex bars start in 2023). Conditional volatility =
   sqrt(h x s x q). Fitted separately on each NSE expiry regime of the sample, as the connectedness study does:
   `single_weekly_thu` (21 Nov 2024 to 29 Aug 2025, 192 days) and `single_weekly_tue` (1 Sep 2025 to 2 Nov 2026,
   288 days), each index on its own. Each series is divided by its median over the sample (this does not change any
   connectedness number). Because each day's VAR is fitted on that day alone, the day's own variance level h cancels;
   the per-day outcome depends on the diurnal shape and the intraday part.
4. **Per-day VAR and connectedness.** For each day d, a VAR(2) with intercept on that day's 360 minutes of
   (Nifty, Sensex) conditional volatility, lags never crossing into another day, OLS; generalized forecast error
   variance decomposition (Pesaran and Shin 1998) at horizon H = 10 minutes, rows normalised to 100
   (`niftygex.connectedness.connectedness_from_fit`, through `connectedness_gamma.day_stats` and `conn_values`). These are
   the settings of the connectedness study's primary test and its per-day regression.
   * **Nifty to Sensex share** = table[Sensex, Nifty] = the share (%) of Sensex's 10-minute forecast error variance
     due to Nifty shocks. **Sensex to Nifty share** = table[Nifty, Sensex].
   * **NET_d** = Nifty to Sensex share minus Sensex to Nifty share, between -100 and +100 percentage points. It equals
     Nifty's Diebold-Yilmaz net directional connectedness for the pair. Positive: Nifty is the net transmitter.
   * **TOT_d** = the sum of the two cross shares (the paper's scale, 0 to 200).
5. **Last hour (secondary window).** The same per-day VAR and decomposition using only the minutes 14:15 to 15:14 (the
   last 60 bars of the session; the volatility series are the whole-day ones). A day enters only if at least 50 of the
   60 minutes are valid for both indices.
6. **Expiry status of a day.** "N" if only Nifty expires (bhavcopy), "S" if only Sensex expires (rule), "none",
   or "both". Holiday-shifted expiries count as expiry days on the day they happened.
7. **Week and period.** A week is the calendar week starting on Monday. A day is "after" if its week starts on or after
   the split date (all split dates are Mondays: 1 Sep 2025 for H1).
8. **Cells.** A day enters a cell (weekday x period) only if its expiry status is the one the design expects for that
   cell (table in section 5). Days on a cell weekday with a different status are left out and listed
   (`results/sensex_swap/excluded_days.csv`).
9. **A day whose VAR cannot be fitted** (singular matrices) gives NaN and is left out of that analysis and counted.

## 5. Sample windows and cells (REAL DATA counts, `results/sensex_swap/coverage.csv`)

Each cell has one day per week, so its number of days is also its number of weeks. "Monthly" counts days that are a
monthly expiry of either index (Sensex monthly taken as the last Sensex expiry of each calendar month, an
approximation). "Auction" counts days from 3 August 2026.

**H1 and S1 (window 1 Jan 2025 to 2 Nov 2026, split 1 Sep 2025).** No Tuesday or Thursday in the window was left out.

| Cell | Expected expiry | Days | First | Last | Monthly | Auction |
| --- | --- | --- | --- | --- | --- | --- |
| Tuesday, before | Sensex only | 34 | 7 Jan 2025 | 26 Aug 2025 | 8 | 0 |
| Thursday, before | Nifty only | 33 | 2 Jan 2025 | 28 Aug 2025 | 8 | 0 |
| Tuesday, after | Nifty only | 56 | 2 Sep 2025 | 27 Oct 2026 | 13 | 12 |
| Thursday, after | Sensex only | 56 | 4 Sep 2025 | 29 Oct 2026 | 11 | 13 |

Without auction days (robustness): Tuesday after 44 days, Thursday after 43 (to 28 and 30 Jul 2026).

**H3 (window 21 Nov 2024 to 31 Aug 2025, split 6 Jan 2025, the week of the first Tuesday Sensex expiry).** Nifty
expires on Thursday throughout.

| Cell | Expected expiry | Days | First | Last |
| --- | --- | --- | --- | --- |
| Tuesday, before | none | 6 | 26 Nov 2024 | 31 Dec 2024 |
| Friday, before | Sensex only | 7 | 22 Nov 2024 | 3 Jan 2025 |
| Tuesday, after | Sensex only | 34 | 7 Jan 2025 | 26 Aug 2025 |
| Friday, after | none | 31 | 10 Jan 2025 | 29 Aug 2025 |

The window starts on 21 Nov 2024 (the first day of NSE's single-weekly regime) because until 19 Nov 2024 NSE had
several weekly index expiries, including Fin Nifty on Tuesday (`data/external/other_weekly_expiries.csv`), so Tuesday
was not a no-expiry day.

**Placebo PL1 (Monday and Wednesday, window and split as H1, only days on which no index expires).**
Monday before 32, Wednesday before 31 (Nifty's shifted 9 and 30 Apr 2025 left out), Monday after 55 (Nifty's five
shifted Mondays left out), Wednesday after 54 (Sensex's five shifted Wednesdays left out).

**Placebos PL2 and PL3 (Tuesday and Thursday at fake swap dates, cells keep the same expiry status on both sides).**
PL2: 1 Jan 2025 to 31 Aug 2025, split 5 May 2025: Tuesday 17 / 17, Thursday 16 / 17 days (before / after). PL3:
1 Sep 2025 to 31 Jul 2026 (before the auction), split 16 Feb 2026: Tuesday 23 / 21, Thursday 21 / 22. Each split sits
near the middle of its window, chosen by week counts only.

**H2 (all 480 days, 21 Nov 2024 to 2 Nov 2026).** 204 expiry days of either index (including the 12 holiday-shifted
ones) and 276 other days. By Sensex period: Friday period (to the week of 30 Dec 2024) 14 expiry days (7 Nifty,
7 Sensex) and 17 others; Tuesday period (6 Jan to 29 Aug 2025) 68 (2 shifted) and 93; Thursday period (from 1 Sep 2025)
122 (10 shifted) and 166.

## 6. Primary test and decision rule

### 6.1 Statistic

delta from section 2, computed from the four cell means of NET_d (whole day). It equals the interaction coefficient of
the day-level regression NET_d = a + b Tue_d + c After_d + delta (Tue_d x After_d) + e_d on the 179 cell days.

### 6.2 Confirmatory p-value: block permutation of the period label over weeks

* The weeks that contain at least one cell day are put in time order. Each has a label: before or after.
* One draw: cut the sequence of weeks into blocks of **4 consecutive weeks** (the first cut at a random offset), put the
  blocks in a random order, and give week w the label of the week now at its position
  (`connectedness_gamma.block_permutation`). The Tuesday and the Thursday of a week always get the same label, and a
  Tuesday always stays a Tuesday. delta is recomputed from the cell means with the new labels.
* **9,999 draws**, seed `numpy.random.default_rng([20261102, 92, 1])` (20261102 is the project's master seed, 92 this
  study). One-sided p = (1 + number of draws with delta* >= delta) / (1 + number of usable draws). A draw with an empty
  cell is not usable and is counted.
* **Why this is primary.** It needs no normality and no large-sample formula, and it is exact if blocks of 4 weeks are
  exchangeable when the swap has no effect. Moving both weekdays of a week together keeps whatever the two days of
  a week share (a calm or a stormy week), and the 4-week blocks keep slow persistence in NET. The HAC regression is
  reported next to it, but this project's event study found HAC p-values alone to reject too often in before-and-after
  comparisons of weekly data (`docs/EVENT_STUDY.md`, section 6; as summarised in `docs/SENSEX_CONFOUND.md`, section 4).
* **Reported with it, not decisive:** the HAC version (the regression in 6.1 with Newey-West standard errors, 5 lags,
  days in date order; one-sided p from the normal distribution), its 95% interval, and its one-sided 95% upper bound
  delta + 1.645 x se. If the permutation p and the HAC p fall on different sides of 0.05, the write-up says so.

### 6.3 Decision rule (level 0.05)

* **"H1 supported"** if delta > 0 and the one-sided permutation p < 0.05.
* **"Evidence against H1"** if delta < 0 and the lower-tail permutation p, (1 + #{delta* <= delta}) / (1 + draws),
  is below 0.05: on its own expiry day the expiring index is the net *receiver*.
* **"H1 not supported"** otherwise, in one of two forms:
  * **"effects as large as the pre-registered minimum detectable effect are ruled out"** if the HAC one-sided 95%
    upper bound is below **4.90 points** (section 6.4);
  * **"inconclusive"** if it is not.
* **Qualifiers from the placebos** (section 8), added to the verdict whatever it is: "design check failed" if PL2 or
  PL3 has two-sided p < 0.05; "the shift is not confined to the expiry day itself" if PL1 has p < 0.05 with the same
  sign as delta; "a warning about the design" if PL1 has p < 0.05 with the opposite sign. A supported H1 with a failed
  design check is to be read as unreliable.

The function `primary_verdict` in the script implements these rules and is tested.

### 6.4 Minimum detectable effect (from the design, before any outcome)

With n_1 to n_4 days in the four cells and day-to-day standard deviation sigma of NET within a cell, the standard error
of delta is sigma x sqrt(1/n_1 + 1/n_2 + 1/n_3 + 1/n_4) if days are independent. The smallest delta detected with 80%
power by a one-sided test at 0.05 is (1.645 + 0.842) x that standard error.

* **Cells 34, 33, 56, 56: MDE = 0.768 sigma.**
* **The value of sigma is an assumption, not estimated from the Nifty and Sensex outcome.** It is taken from the Nifty
  and Bank Nifty connectedness study's published per-day file (`results/connectedness_gamma/sample_days.csv`, columns
  `day_nifty_to_bn` minus `day_bn_to_nifty`, same volatility input, VAR(2) and horizon): in its two most recent regimes
  (21 Nov 2024 to 22 Apr 2026, 348 days) the standard deviation of that NET after removing each regime's weekday means
  is **sigma = 6.38 points** (`results/sensex_swap/mde_proxy.csv`).
* **Pre-registered MDE for H1: 4.90 points of delta.** Equivalently, with one common effect, a Nifty-expiry day and a
  Sensex-expiry day on the same weekday must differ in NET by about 2.45 points (theta = 1.23 points per expiry).
* Sensitivity, for reading only: 5.57 points if same-weekday residuals one week apart are correlated as in the proxy
  (0.13, so the variance of a cell mean is inflated by (1 + 0.13) / (1 - 0.13)); 2.45 points if sigma is half the
  proxy and 9.80 if it is twice. Nifty and Sensex are much closer to each other than Nifty and Bank Nifty, so their
  NET could vary less or more from day to day; the MDE in sigma units (0.768) holds either way. In the proxy, the
  Tuesday and Thursday of the same week are nearly uncorrelated (0.03), so independence is a fair working assumption.

The 4.90 points is fixed here (the constant `PREREG_MDE` in the script) and is the number used in the decision rule
even if the observed spread of NET, or a regenerated proxy file, turns out different. The write-up also reports the MDE
implied by the observed within-cell spread (`verdict.json`), labelled as not decisive.

## 7. Secondary hypotheses (Holm-corrected as one family of three)

| Test | Hypothesis | Statistic and p-value | MDE (80% power) |
| --- | --- | --- | --- |
| **S1** | H1 in the last hour (14:15 to 15:14) | delta from per-day last-hour NET; one-sided block-permutation p as in 6.2 (seed key 2) | 0.77 sigma_LH at 0.05, 0.92 sigma_LH at 0.05/3; no published last-hour per-day values exist, so no value in points (sigma_LH is expected to be larger than the whole-day sigma because each VAR has 60 instead of 360 minutes) |
| **S2** | H2: total connectedness differs on either index's expiry day from other days (CPP's question on a new pair) | Day-level OLS of TOT_d on an expiry dummy (Nifty or Sensex expires, including holiday-shifted days), weekday dummies and Sensex-period dummies (Friday, Tuesday, Thursday period), all 480 days; Newey-West 5 lags; **two-sided** p | 3.06 points at 0.05, 3.53 at 0.05/3 (sigma_TOT = 5.27 from the same proxy file) |
| **S3** | H3: the same NET logic at the first event, Sensex moving from Friday to Tuesday (week of 6 Jan 2025) while Nifty stays on Thursday | delta3 = [NET(Tue, after) - NET(Tue, before)] - [NET(Fri, after) - NET(Fri, before)]; **one-sided, delta3 < 0** (Sensex becoming the net transmitter on its new day lowers NET there); block permutation as in 6.2 (seed key 3) | 9.67 points at 0.05, 11.55 at 0.05/3 (cells 6, 7, 34, 31: low power) |

* **H2 identification.** With weekday and period effects in the regression, the expiry dummy is identified only by
  days whose expiry status differs from that of the same weekday in another period: Tuesday and Friday in the Friday
  period (6 and 7 days) against the later periods, and the 12 holiday-shifted expiry days against ordinary Mondays and
  Wednesdays. Without weekday effects, "expiry day" would mostly mean "Tuesday or Thursday". A version without weekday
  effects is reported as description only (robustness file).
* **Holm.** The three p-values (S1 and S3 permutation, S2 HAC) are adjusted with Holm's step-down method
  (`connectedness_gamma.holm`). A secondary hypothesis is "supported" if its Holm-adjusted p < 0.05 and, for S1 and
  S3, the estimate has the stated sign. Secondaries cannot change the primary verdict.

## 8. Placebos and design checks (two-sided, not corrected)

Placebos are not corrected for multiple testing: a placebo exists to catch a problem, so it is judged at the
stricter, uncorrected 0.05. Each uses the block permutation of 6.2 (two-sided p = (1 + #{|delta*| >= |delta|}) /
(1 + draws)) and reports the HAC p as well.

* **PL1, Monday against Wednesday around 1 September 2025** (seed key 4), days on which no index expires. No expiry
  moved on to or away from these days. **Caveat, stated now:** the swap also changed which index expires *the next
  day*: Monday was the eve of a Sensex expiry before and of a Nifty expiry after, and Wednesday the other way round. If
  the expiring index already leads on the eve of its expiry, PL1 moves in the same direction as H1. So a significant PL1
  with the sign of delta means "the effect is not confined to the expiry day", not that the design is broken; a
  significant PL1 with the opposite sign is a warning.
* **PL2, a fake swap on 5 May 2025** inside the before window (seed key 5), and **PL3, a fake swap on 16 Feb 2026**
  inside the after window, stopping before the auction (seed key 6). Tuesday against Thursday, nothing changed on these
  dates. A significant PL2 or PL3 means the Tuesday-minus-Thursday gap drifts on its own, which breaks the identifying
  assumption: **"design check failed"**.

## 9. Robustness of the primary (reported whatever they show; never decisive; no correction)

Each is the primary statistic with one change, with its own permutation seed (keys in `run_info.json`):
block lengths of 1, 2 and 8 weeks; labels moved separately within each weekday (blocks of 4 days of the same weekday,
so a week's Tuesday and Thursday can get different labels); closing-auction days left out; monthly expiry days of
either index left out; the intraday mcsGARCH part sqrt(q) as input (no daily level, no diurnal shape); VAR lag chosen
by AIC (maximum 10) on all sample days with horizon 30 (the connectedness study's sensitivity setting); one mcsGARCH
fit over the whole sample instead of one per regime; week fixed effects (Tuesday minus Thursday within each week that
has both, then after minus before); ranks of NET instead of values (outliers); each half of NET separately (the Nifty
to Sensex share, expected to rise, and the Sensex to Nifty share, expected to fall); total connectedness TOT_d
(description); a CPP-style version that pools all days of a cell into one VAR and takes NET from the pooled fit.

## 10. Treatment of holiday-shifted expiries and of the closing auction

* **Holiday-shifted expiries.** When an expiry is moved to the previous trading day, that day is the expiry day. In
  the sample all 12 shifts land on a Monday or Wednesday, so they never enter the H1, S1 or H3 cells; the week keeps its
  other cell day. They count as expiry days in H2 (they are part of what identifies it), and they are left out of the
  placebo PL1. A Tuesday or Thursday whose status differs from the expected one (for example both indices expiring)
  would be left out; there is none in the sample.
* **Closing auction from 3 August 2026.** In the bars, both indices stand still from 15:16 to 15:28 on about 96% of
  those minutes from 3 Aug 2026 (Nifty 96.5%, Sensex 96.3%; under 1% before), so BSE appears to have started a closing
  auction on the same day as NSE (seen in the data, not checked against a BSE circular). The 360-bar session ends at
  15:14, so the frozen minutes and the auction bar never enter any VAR. The auction can still change trading before
  15:15 on expiry days (the settlement price is set differently), and it affects both Tuesday and Thursday in the after
  window. The primary keeps these days, because the window was fixed by the project lead; the robustness check ends the
  after window on 31 Jul 2026 (Tuesday 44 and Thursday 43 days). If the two disagree about the 0.05 line, the write-up
  says that the result depends on the auction period. PL3 stops before the auction. The mcsGARCH fit of NSE's
  `single_weekly_tue` regime (1 Sep 2025 to 2 Nov 2026) includes the auction days.

## 11. What is reported whatever happens

* The verdict of 6.3 with its qualifiers, and the primary delta with the permutation p, the HAC p and the HAC 95%
  interval, in the first lines of the write-up.
* For every test: the four cell means, standard deviations, medians and day counts of NET (and of the two shares and
  TOT behind them), the estimate, permutation p, the mean and spread of the permutation draws, HAC p and interval, and
  the Holm p for the secondaries. All placebos and every robustness row, including those that disagree.
* The per-day file with every NET, share and TOT value, the mcsGARCH fits and the calendar flags; the MDE table and
  the MDE implied by the observed spread.
* Files in `results/sensex_swap/`: `calendar_days.csv`, `calendar_check.csv`, `coverage.csv`, `excluded_days.csv`,
  `mde.csv`, `mde_proxy.csv`, `run_info_coverage.json` (phase 1), and `day_values.csv`, `cell_means.csv`, `tests.csv`, `robustness.csv`,
  `h2_regression.csv`, `mcsgarch_fits.csv`, `verdict.json`, `run_info.json` (phase 2); logs
  `results/sensex_swap_coverage_stdout.txt` and `results/sensex_swap_stdout.txt`; write-up `docs/SENSEX_SWAP.md`.
* A null or negative primary result is written up with the same prominence as a positive one.

## 12. Deviations policy

* Any change after the commit of this file (definitions, windows, cells, inference, decision rule, code that changes a
  number) is listed in `docs/SENSEX_SWAP.md` with the date, the reason and, whenever possible, both the pre-registered
  and the changed result side by side. **The pre-registered version stays the primary result.**
* Pure bug fixes that make the code do what this document says are listed too, with the size of their effect.
* New data after 2 Nov 2026 are not added to these tests. A rerun on later data would be a new, separately
  pre-registered study.
* If a primary BSE or SEBI document later shows a Sensex expiry date in the sample to be wrong, the analysis is rerun
  with the corrected calendar and both versions are reported; the corrected one is labelled as a deviation.
* Analyses not listed here (for example other horizons, other VAR lags, other indices, or splitting by gamma) are
  exploratory, labelled "not pre-registered", and cannot support or contradict H1.

## 13. Reproducibility

```
OMP_NUM_THREADS=1 python experiments/sensex_swap.py --coverage   # phase 1: calendars and coverage, no outcome
OMP_NUM_THREADS=1 python experiments/sensex_swap.py              # phase 2: the analysis above
OMP_NUM_THREADS=1 python -m pytest -q tests/test_sensex_swap.py  # 18 tests on SYNTHETIC data
```

Seeds: `numpy.random.default_rng([20261102, 92, key])`, key 1 (H1), 2 (S1), 3 (S3), 4 to 6 (PL1 to PL3), 11 to 25
(robustness, listed in `run_info.json`). The script stores the first 16 hex digits of the SHA-256 of each input in
`run_info.json` and checks that one day's connectedness from its fast per-day code equals
`replication.connectedness_segments` on that day. The tests check, on simulated minute data, that a planted swap effect
is detected (permutation p at most 0.002), that weekday and period effects without a swap effect are not
(p above 0.05), that the permutation p-value is roughly uniform when there is no effect, and that the whole pipeline
runs end to end on synthetic files.

## 14. Known weaknesses of the design (stated before the results)

1. **Only the sum of the expiry effects is identified** (section 2), and only under the parallel-gap assumption. PL2 and
   PL3 test that assumption at two dates; they cannot rule out a change that happened exactly on 1 Sep 2025 for some
   reason other than the swap.
2. **The swap moved everything at once**: the weekly index options, the monthly index and single-stock contracts of
   each exchange, and the eves of expiry. "Nifty expiry day" means "NSE expiry day".
3. **NET at the one-minute scale may be small for two near-identical indices.** With strongly correlated shocks, the
   generalized decomposition splits the same-minute co-movement almost evenly, so NET comes mainly from which index's
   volatility predicts the other's in the next minutes. That lead can partly reflect how each exchange computes and
   publishes its index and how the data provider stamps the bars. Such effects cancel in delta as long as they are the
   same on Tuesdays and Thursdays in both periods.
4. **The Sensex bars are not validated against BSE**, and the Sensex calendar is reported, not verified.
5. **Power.** The MDE rests on a borrowed variance. H3 has only 6 and 7 days before its event and can detect only
   large effects.
6. **The after window includes 13 weeks of the closing auction** (section 10).
7. **One-minute volatility is a model output** (mcsGARCH fitted in-sample per regime), as in CPP and our connectedness
   study.

## Sources

* Chhimwal, B., Pandey, V. and Pandey, P. (2025). Effect of multiple index derivative expiry on volatility, volume, and
  connectedness: a tale of two stock indices in India. *Review of Derivatives Research*, 28(3), article 14.
  doi:10.1007/s11147-025-09221-8 (verified in `docs/LITERATURE_REVIEW.md`).
* Engle, R. F. and Sokalska, M. E. (2012). Forecasting intraday volatility in the US equity market: multiplicative
  component GARCH. *Journal of Financial Econometrics*, 10(1), 54 to 83. doi:10.1093/jjfinec/nbr005 (verified in
  `docs/LITERATURE_REVIEW.md`).
* Cited from memory, not looked up for this document: Diebold, F. X. and Yilmaz, K. (2012), "Better to give than to
  receive", *International Journal of Forecasting* 28(1); Pesaran, M. H. and Shin, Y. (1998), "Generalized impulse
  response analysis in linear multivariate models", *Economics Letters* 58(1); Newey, W. K. and West, K. D. (1987),
  *Econometrica* 55(3); Holm, S. (1979), "A simple sequentially rejective multiple test procedure", *Scandinavian
  Journal of Statistics* 6(2).
* Sensex calendar sources: the rows of `data/external/sensex_expiry_calendar.csv` (each with its address and access
  date), including the report that NSE equity derivatives moved to Tuesday and BSE's to Thursday from 1 Sep 2025
  (Moneylife, June 2025).
