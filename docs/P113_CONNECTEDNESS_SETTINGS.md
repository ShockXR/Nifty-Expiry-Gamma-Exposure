# P113: which connectedness setting reproduces the paper's Tables 3 and 4?

Study P113, run by an AI agent (Claude) on 3 November 2026 for Shreyan Thanki's project with
Prof. Piyush Pandey. All numbers below are REAL DATA unless marked **paper (printed)** or SYNTHETIC.
Every number comes from a file in `results/p113_connectedness_settings/` (named next to each table) or
from the paper's Tables 3 and 4 as re-keyed blind in `results/replication_nse/rekey_blind.csv`.

* Registered plan: `docs/prereg/P113_CONNECTEDNESS_SETTINGS.md`, commit
  `313a3ac0c1e12529c99ef61d5d2a60060b6f7a75` (09:45:09 UTC, pushed before any connectedness number
  of this study was computed).
* Code: `experiments/p113_connectedness_settings.py` (imports the replication's data and mcsGARCH code
  from `experiments/replicate_paper.py`); new functions `cholesky_fevd` and `fevd_table` in
  `src/niftygex/connectedness.py` (existing functions and defaults unchanged).
* Tests (SYNTHETIC only): `tests/test_p113_connectedness_settings.py`.

## 1. The answer in plain words

**No setting in the registered grid reproduces the paper's 15 printed connectedness totals. Under the
registered rule the verdict is "not reproduced"**, on the primary before window and on both secondary
windows (`verdict.csv`).

* **The grid.** 5 ways of splitting the forecast error variance, 8 input series, 8 forecast horizons,
  and 17 combinations of how the sub-sample is built and how many VAR lags are used: 5,440 settings per
  before window, 16,320 with the three before windows.
* **The best setting** (column-normalised generalized decomposition, log of the mcsGARCH conditional
  variance, sub-sample days joined end to end, VAR(5), horizon 5) misses the printed totals by
  **5.34 points on average**, with **8 of 15 within 2 points**. It is clearly better than the
  replication's main setting (9.26) and better than chance (shuffle test p = 0.003), but it misses the
  before last hour by 23 points (82.08 against 105.22) and the after Bank Nifty first hour by 21 points.
  The row-normalised version of the same setting (the standard Diebold and Yilmaz 2012 table) is a near
  tie (5.37), so the data cannot tell row from column normalisation.
* **The total of 105.22 (before, last hour) cannot be reached by any normalised or Cholesky setting on
  our data.** Over all 4,352 such settings the largest total in that cell is 94.66 on window A (94.06 on
  window B, 87.48 on the stated window), and the largest smaller cross share is 43.80% (the paper prints
  52.79% and 52.43%). In that cell only the unnormalised generalized shares (Pesaran and Shin 1998) go
  above 100, and they go above 100 almost everywhere: for every one of the 15
  cells, between 433 and 1,054 of the 1,088 unnormalised settings give a total above 100, while the paper
  prints only one total above 100. The best unnormalised setting misses by 17.95 points on average.
* **The other two outlying totals can each be reached, but not together with the rest.** 99.42 (before,
  first hour) is within 2 points for 875 settings and 80.18 (after, Nifty first hour) for 125 settings,
  but no setting brings all three outlying cells within 5 points while missing the 15 totals by 5 points
  or less on average.
* **Even the after cells are not reproduced.** Our after sample matches the paper's Table 1 to the
  printed digits, so the right setting should reproduce the 10 after totals closely. The best setting
  misses them by 4.20 points on average, and this is no better than the grid does when the 10 after
  totals are shuffled among the after cells (p = 0.63). So the gap is not only about the uncertain
  before window: the paper's tables were built in some way that none of the 5,440 settings captures.
* **The paper's written method** (normalised generalized decomposition on the mcsGARCH conditional
  variance or volatility) does no better: its best setting misses by 7.17 points on average, 2 of 15
  within 2 points. On the conditional variance itself, which is what the paper's Section 3 says it uses,
  the best is 9.26.
* **The paper's conclusions, under the best setting:** connectedness does fall on expiry days after the
  change (by 7.95 and 10.00 points, both with day-level bootstrap intervals that exclude zero), but the
  fall is **not** larger in the last hour (the last-hour changes are -8.86 and -6.35, with intervals
  that include zero, against printed -38.78 and -30.79).

**The question for the authors (Section 5):** were the shares in Tables 3 and 4 normalised so that each
row sums to 100, and with which software, input series, VAR lag and horizon? In the last hour before the
change both printed cross shares are above 50%, which no normalised or Cholesky setting produces on our
data, while unnormalised shares put almost every other cell above 100 too.

## 2. What was done

The paper (Section 3.2) names Diebold and Yilmaz (2012), writes the generalized variance decomposition
and a normalisation, and does not name a package, a VAR lag or a horizon. Its text says the
normalisation concerns "columns" while Diebold and Yilmaz normalise rows. Section 3 says the
decomposition is applied to the mcsGARCH conditional variance. So the grid treats the normalised
generalized decomposition on the conditional variance as the main candidate and adds the departures
the task listed (plan, Sections 2 and 6):

| Dimension | Values (code) |
|---|---|
| Decomposition (5) | generalized, rows normalised (GR, Diebold and Yilmaz 2012); generalized, not normalised (GU); generalized, columns normalised (GC); Cholesky with Nifty first (CN) or Bank Nifty first (CB) |
| Input series (8) | mcsGARCH conditional variance (cvar), its square root (cvol), its log (logvar), the mcsGARCH intraday part (qvol), 1-minute returns (ret), absolute returns (absret), squared returns (sqret), Rogers-Satchell volatility per bar (rsvol) |
| Horizon H (8) | 1, 2, 5, 10, 11, 20, 50, 100 steps |
| Construction and VAR lag (17) | segmented, lags never cross a day or hour block (S), lags 1, 2, 3, 5, 10, AIC, BIC; concatenated, blocks joined end to end (C), the same 7 lags; rolling 200-minute VAR averaged over the minutes of the sub-sample (R), lags 1, 2, 5 |
| Before window (3) | A: 14 November 2022 to 10 February 2023 (primary, the replication's); B: 22 November 2022 to 20 February 2023 (P108); ST: 1 November 2023 to 29 February 2024 (stated in the paper) |

After window: 1 March to 31 May 2024. Everything else (360 minutes a day from 09:15 to 15:14, the hour
definitions, the expiry calendar, mcsGARCH) is as in `docs/REPLICATION_NSE.md`.

**Baseline check passed.** The replication's main setting (GR, cvol, S, lag 2, H = 10) reproduces every
cell of `results/replication_nse/tables34_connectedness.csv` exactly (largest difference in a total 0,
in a cross share 7e-15; same days and observation counts; `baseline_check.csv`).

**Fit criterion (registered).** MAD15 = mean absolute difference between our total and the printed
total over the 15 cells, on the paper's scale (sum of the two cross shares). K2 = number of cells within
2 points. Guards: a shuffle test (the 15 printed totals reassigned to the wrong cells 1,000 times, the
grid searched again each time) and leave-one-table-out (choose the best setting on two tables, score it
on the third).

## 3. Results (window A unless stated)

### 3.1 The best settings, cell by cell

From `spec_fit.csv` and `windows.csv`. Codes as in Section 2 (decomposition | input | construction |
lag | horizon).

| Cell | Paper (printed) | Replication's main setting: GR, cvol, S, 2, 10 | Best: GC, logvar, C, 5, 5 | Best of the written method: GR, cvol, S, 10, 100 | Best on cvar: GR, cvar, S, 1, 2 |
|---|---|---|---|---|---|
| T3.1 before, all expiry days | 84.00 | 87.94 | 84.43 | 82.88 | 92.80 |
| T3.2 before, monthly | 74.70 | 81.09 | 82.11 | 80.69 | 77.93 |
| T3.3 before, first hour | **99.42** | 81.59 | 94.24 | 90.28 | 89.10 |
| T3.4 before, middle hour | 68.64 | 78.81 | 66.68 | 74.46 | 65.27 |
| T3.5 before, last hour | **105.22** | 84.73 | 82.08 | 87.11 | 77.07 |
| T4.I.1 after, Nifty expiry days | 76.38 | 74.89 | 76.48 | 74.23 | 78.09 |
| T4.I.2 after, Nifty monthly | 74.58 | 67.66 | 74.49 | 68.12 | 67.20 |
| T4.I.3 after, Nifty first hour | **80.18** | 67.59 | 88.74 | 66.60 | 71.34 |
| T4.I.4 after, Nifty middle hour | 68.81 | 50.21 | 69.24 | 64.90 | 44.03 |
| T4.I.5 after, Nifty last hour | 66.44 | 77.44 | 73.21 | 82.30 | 70.34 |
| T4.II.1 after, Bank Nifty expiry days | 73.02 | 68.92 | 74.43 | 66.74 | 68.98 |
| T4.II.2 after, Bank Nifty monthly | 72.74 | 74.13 | 74.97 | 69.97 | 75.95 |
| T4.II.3 after, Bank Nifty first hour | 70.97 | 81.01 | 91.84 | 80.13 | 80.47 |
| T4.II.4 after, Bank Nifty middle hour | 66.54 | 58.43 | 66.26 | 67.44 | 52.34 |
| T4.II.5 after, Bank Nifty last hour | 74.43 | 80.33 | 75.73 | 80.80 | 81.96 |
| **MAD15** (mean miss, points) | | 9.26 | **5.34** | 7.17 | 9.26 |
| **K2** (cells within 2 points) | | 2 | **8** | 2 | 1 |
| Rank among 5,440 | | 415 | 1 | 87 | 416 |

The best setting fits the three all-expiry rows within 1.5 points (84.43 against 84.00 before, 76.48
against 76.38 and 74.43 against 73.02 after). Its cross shares are close to
the printed ones on average (mean miss 3.03 points over the 30 printed cross shares). Its weak point is
the first and last hours: it gives high first-hour totals in all three periods (94.24, 88.74 and 91.84),
so it cannot reproduce the paper's pattern of a very high first hour before the change and a lower one
after, and its before last hour (82.08) is far from 105.22.

**Near-ties** (`near_ties.csv`): six settings lie within 0.5 points of the best (MAD15 5.34 to 5.84).
All six use the log conditional variance, the concatenated construction and a lag of 5 or chosen by BIC,
with H = 2 or 5. They come in pairs, GC and GR, that differ by less than 0.04 points, because our tables
are nearly symmetric and then row and column normalisation give almost the same total. So the data point
to "log conditional variance, days joined end to end, short horizon" but cannot separate row from column
normalisation.

**By dimension** (`by_dimension.csv`, best MAD15 among the settings with that value): GR 5.37, GC 5.34,
CB 8.87, CN 10.54, GU 17.95; logvar 5.34, cvol 7.17, ret 8.45, absret 8.89, rsvol 9.14, cvar 9.26,
qvol 11.59, sqret 12.24; concatenated 5.34, segmented 6.67, rolling 7.97.

### 3.2 Can any setting produce a total above 100?

From `reach.csv`: for each decomposition, the largest total and the largest smaller cross share over its
1,088 settings, in the three outlying cells. A total above 100 needs the smaller cross share above 50.

| Window, cell (printed total; printed smaller cross share) | GR | GC | CN | CB | GU |
|---|---|---|---|---|---|
| A, before first hour (99.42; 49.67) | 100.00; 49.97 | 100.00; 50.00 | 100.11; 43.70 | 100.00; 17.53 | 199.77; 99.89 |
| A, before last hour (105.22; 52.43) | **91.05; 43.65** | **89.99; 43.80** | **94.66; 42.56** | **94.64; 36.50** | 150.82; 75.10 |
| B, before last hour | 90.40; 44.17 | 89.33; 44.10 | 94.06; 42.81 | 91.35; 35.32 | 154.83; 75.27 |
| ST, before last hour | 87.48; 41.08 | 87.35; 41.56 | 87.08; 17.61 | 74.87; 20.22 | 139.23; 67.22 |
| after, Nifty first hour (80.18; 40.03) | 99.87; 49.93 | 99.87; 49.93 | 99.73; 26.12 | 101.74; 31.42 | 199.45; 99.73 |

Number of settings within 2 points of the printed total in the before last hour (window A): 0 for GR,
GC, CN and CB; 39 for GU. In the before first hour: GR 269, GC 269, CN 152, CB 178, GU 7.

So on our data **the before last-hour cell is out of reach of every normalised and every Cholesky
setting, in all three before windows.** Under the normalised generalized decomposition a total close to
100 (both shares close to 50) is possible, for example in the before first hour, but not above it in that
cell. The Cholesky totals of just over 100 come with very unequal shares (smaller share at most 43.70%),
unlike the paper's nearly equal ones.

**The unnormalised shares overshoot everywhere.** Between 433 (T4.II.5) and 1,054 (T4.I.1) of the 1,088
GU settings give a total above 100 in each of the 15 cells, and 1,037 do so in the before all-expiry cell,
where the paper prints 84.00. Read from `spec_fit.csv` after the run: the GU setting closest to the three
outlying cells (GU, cvar, S, lag 10, H = 2) is within 3.8 points of all three, but misses the after last
hours by 71.3 and 84.5 points (MAD15 21.0). There is also a structural reason against GU: with
unnormalised shares the own share is not 100 minus the cross share, yet in 28 of the paper's 30 printed
rows own plus cross is exactly 100.00 (the other two, in T4.I.3, sum to 101.00).

### 3.3 Guards against fitting a large grid

| Guard (window A) | Result | File |
|---|---|---|
| Shuffle test of the best MAD15 | best 5.34; median best over 1,000 shuffles 7.88; p = 0.003 | `permutation.csv` |
| Shuffle test of the after-only fit | best MAD over the 10 after cells 4.20; median best over shuffles 3.96; p = 0.63 | `permutation.csv` |
| Outlying cells (criterion b) | no setting meets it; in no shuffle does a setting meet it | `permutation.csv` |
| After cells reproduced (criterion c) | no setting meets it | `permutation.csv` |
| Leave one table out | held out: Table 3 7.63, Panel I 7.32, Panel II 5.22; pooled held-out MAD 6.72 (in-sample best 5.34; replication's main setting 9.26) | `loto.csv` |

So the best setting captures something real about the day-level cells (it is better than chance and
holds up out of sample), but not the hour-level cells, and not the after table on its own.

### 3.4 The registered decision rule

From `verdict.csv`. "Reproduced" needs MAD15 <= 2, K2 >= 12, the three outlying cells within 2 points,
held-out MAD <= 3 and shuffle p <= 0.05. "Partly reproduced" needs one of (a) MAD15 <= 4 with K2 >= 8,
held-out MAD <= 5 and shuffle p <= 0.05; (b) a setting with all three outlying cells within 5 points and
MAD15 <= 5; (c) a setting with the 10 after cells within 1 point on average and all within 2.

| Analysis | Best setting | MAD15 | K2 | Largest miss on the outlying cells | Held-out MAD | Shuffle p | Verdict |
|---|---|---|---|---|---|---|---|
| **window A (registered)** | GC, logvar, C, 5, 5 | 5.34 | 8 | 23.14 | 6.72 | 0.003 | **not reproduced** |
| window B | GC, logvar, C, BIC, 5 | 5.28 | 6 | 19.23 | 7.19 | 0.003 | not reproduced |
| window ST (stated) | GC, logvar, C, BIC, 11 | 6.07 | 5 | 25.25 | 6.82 | 0.031 | not reproduced |
| window as a dimension (16,320) | GC, logvar, C, BIC, 5, window B | 5.28 | 6 | 19.23 | 6.91 | 0.006 | not reproduced |
| (i) paper's written method (224) | GR, cvol, S, 10, 100 | 7.17 | 2 | 18.11 | 8.17 | 0.038 | not reproduced |
| (ii) all GR (1,088) | GR, logvar, C, 5, 5 | 5.37 | 8 | 23.14 | 7.22 | 0.003 | not reproduced |
| GU only (1,088) | GU, absret, S, AIC, 1 | 17.95 | 0 | 13.88 | 30.91 | 0.28 | not reproduced |
| CN only / CB only | CN, cvol, R, 5, 100 / CB, logvar, C, 10, 20 | 10.54 / 8.87 | 4 / 2 | 40.42 / 21.96 | 14.21 / 15.33 | 0.54 / 0.099 | not reproduced |
| segmented / concatenated / rolling only | | 6.67 / 5.34 / 7.97 | 4 / 8 / 4 | | 8.96 / 5.37 / 8.08 | 0.056 / 0.001 / 0.21 | not reproduced |

The best setting fails rule (a) only on its mean miss (5.34 against 4.0) and its held-out MAD (6.72
against 5.0). No setting comes near rules (b) or (c).

**Printed slips (S5, `alt_targets.csv`).** With 78.17 for T4.I.3 and 71.97 for T4.II.3 (Section 4 of the
plan) the best setting is the same and its MAD15 is 5.41. The slips do not explain anything.

## 4. Secondary results (window A)

### 4.1 Changes, before to after (S1, `changes.csv`)

| Sub-sample | Panel | Best setting | Written method's best | Paper (printed) |
|---|---|---|---|---|
| all expiry days | I (Nifty) | -7.95 | -8.65 | -7.62 |
| all expiry days | II (Bank Nifty) | -10.00 | -16.14 | -10.98 |
| monthly | I | -7.62 | -12.57 | -0.12 |
| monthly | II | -7.14 | -10.72 | -1.96 |
| first hour | I | -5.50 | -23.68 | -19.24 |
| first hour | II | -2.40 | -10.15 | -28.45 |
| middle hour | I | +2.56 | -9.55 | +0.17 |
| middle hour | II | -0.42 | -7.01 | -2.10 |
| last hour | I | -8.86 | -4.81 | -38.78 |
| last hour | II | -6.35 | -6.31 | -30.79 |

The best setting gets the sign of all 10 printed changes right, 3 within 2 points, mean miss 11.22
points. The written method's best: 9 signs right, 1 within 2 points, mean miss 12.32. Both reproduce the
all-expiry falls; neither reproduces the large one-hour falls.

### 4.2 Direction of spillover (S2, `direction.csv`)

Net spillover of Nifty (Bank Nifty from Nifty minus Nifty from Bank Nifty) has the printed sign in 8 of
15 cells under the best setting and 11 of 15 under the written method's best. Most printed net values
are small (between -3.19 and +1.45 points, except +8.10 in the after Nifty last hour), so signs flip
easily. Over all days, Nifty is a net receiver before the change (net -2.21 under the best setting,
-22.27 under the written method's best) and close to balanced after (-0.31 and -2.39). The paper says
Nifty is a net receiver and that this becomes stronger after the change; under both settings it becomes
weaker, as the replication also found.

### 4.3 The paper's conclusions, with day-level bootstrap intervals (S3, `bootstrap.csv`, `tests.csv`)

999 draws per cell, whole days (or whole hour blocks) resampled; Holm adjustment over the six statements
of each setting.

| Statement | Best setting: change [95% interval], Holm p | Written method's best: change [95% interval], Holm p | Paper (printed) |
|---|---|---|---|
| 1. Falls on Nifty expiry days (all expiry days, Panel I minus Table 3) | -7.95 [-10.00, -5.54], 0.012 | -8.65 [-15.29, -1.96], 0.090 | -7.62 |
| 2. Falls on Bank Nifty expiry days (Panel II minus Table 3) | -10.00 [-12.19, -7.57], 0.012 | -16.14 [-21.35, -9.20], 0.012 | -10.98 |
| 3. Last hour, Panel I minus Table 3 | -8.86 [-14.09, 1.68], 0.53 | -4.81 [-13.06, 11.05], 0.79 | -38.78 |
| 4. Last hour, Panel II minus Table 3 | -6.35 [-12.47, 3.27], 0.53 | -6.31 [-18.84, 11.43], 0.50 | -30.79 |
| 5. Last-hour change minus all-expiry change, Panel I | -0.92 [-7.02, 9.56], 0.82 | +3.85 [-6.48, 20.20], 0.79 | -31.16 |
| 6. The same, Panel II | +3.65 [-3.36, 13.85], 0.66 | +9.84 [-4.65, 25.85], 0.77 | -19.81 |

So "connectedness falls on expiry days after the change" holds under both settings (statement 2 under
both, statement 1 under the best setting; under the written method's best, statement 1 has an interval
that excludes zero but a Holm p of 0.090). "The fall is largest in the last hour" does not hold under
either: the last-hour falls are of the same size as the all-expiry falls, and their intervals include
zero. Each one-hour cell has only three hours of data, so these intervals are wide and coarse.

## 5. The question for the authors

> Were the variance shares in Tables 3 and 4 normalised so that each row sums to 100 (as in Diebold and
> Yilmaz 2012), and with which software, input series, VAR lag and forecast horizon? We ask because in the
> last hour before the change both cross shares are above 50% (52.79 and 52.43), which none of our 4,352
> normalised or Cholesky settings produces on our data (the largest total there is 94.7), while
> unnormalised shares, which can exceed 100, put almost every other cell above 100 as well.

Two follow-ups, if the authors have time: were the expiry days (or hours) joined end to end before the
VAR was fitted (our closest settings do this), and how were the "first", "middle" and "last" trading
hours defined (we used 09:15 to 10:14, 11:45 to 12:44 and 14:15 to 15:14)?

## 6. Deviations and disclosures

* **No deviation from the registered analysis.** Every setting, criterion, guard and secondary analysis
  is as in the plan.
* **A debugging run.** Before the registered run, the same code was run once with 20 shuffles and 20
  bootstrap draws (instead of 1,000 and 999) to catch errors. Its outputs were overwritten by the
  registered run. No code or setting was changed between the two runs, apart from the run-time note in
  the script's docstring, which was corrected afterwards (it said about 15 minutes; the run takes about
  2.5 minutes).
* **Two primary rows of `tests.csv` have no p-value.** The plan defines the shuffle p-value for a claim
  of the form "some setting achieves criterion X". For criteria (b) and (c) no setting achieves them, so
  no claim is made and p is left empty (NaN), as the plan's code path specifies. The rows are kept so the
  ledger shows that these tests were registered. For the after-only fit, the shuffle p of the best
  MAD over the 10 after cells (0.63) is given in the row's text and in `permutation.csv`.
* **What was read after the run, not registered:** the GU setting closest to the outlying cells
  (Section 3.2) and the row sums of the printed tables are descriptive readings of the registered outputs
  and of the printed numbers; they add no new setting.

## 7. Caveats

* **The grid is large but finite.** Not tested: other definitions of the first, middle and last hour
  (for example a last hour ending at 15:29), other day sets, the paper's exact mcsGARCH estimates
  (ours estimate the ARMA and GARCH parts separately), time-varying-parameter VARs, other rolling window
  lengths, and conventions of particular software packages beyond the decompositions listed. A setting
  outside the grid could still reproduce the tables.
* **Our before data are not exactly the paper's.** The before window is uncertain (14 to 23 November 2022
  start, P108) and one bar differs on 27 January 2023 (`docs/REPLICATION_NSE.md`). This is why the after
  cells matter: there our data match the paper's Table 1 to the printed digits, and no setting reproduces
  them either.
* **One-hour cells are tiny.** Each one-hour cell has 180 observations from three days. A setting can be
  close in one cell by luck, which is why the registered criterion asks for a single setting across all
  15 cells and adds the shuffle and leave-one-table-out guards. The bootstrap of a one-hour cell resamples
  only three hour blocks, so its intervals are coarse; for the segmented VAR(10) the point estimate sits
  at the edge of several intervals (for example 66.60 [66.60, 85.78] in T4.I.3).
* **The shuffle test is conservative.** Many printed totals are similar (66 to 84), so a shuffle often
  changes the targets little. A p of 0.003 for the best MAD15 is therefore strong evidence that the best
  setting captures real structure, while a large p (as for the after-only fit) says the fit there is no
  better than what the grid's flexibility gives.
* **Row and column normalisation cannot be separated** on these data (Section 3.1). The question for the
  authors asks about normalisation in general.

## 8. Files and how to reproduce

`results/p113_connectedness_settings/` (all REAL DATA except `targets.csv`, which holds the printed values):

| File | What |
|---|---|
| `targets.csv` | the 15 printed totals and 60 shares, from `results/replication_nse/rekey_blind.csv`, checked against `P_T34` |
| `baseline_check.csv` | the replication's main setting against the committed `tables34_connectedness.csv` |
| `grid_cells.csv.gz` | every setting and cell: cross shares, total, lag used, observations |
| `spec_fit.csv` | every setting's MAD15, K2, OUT3, MAD10a, CS30, errors per cell and rank, per window |
| `best.csv`, `near_ties.csv`, `by_dimension.csv` | the top 25, the near-tie set, the best per value of each dimension |
| `verdict.csv`, `subsets.csv`, `windows.csv`, `alt_targets.csv` | the decision rule for each analysis |
| `permutation.csv`, `loto.csv` | the two guards |
| `reach.csv` | largest totals and shares per decomposition and cell |
| `changes.csv`, `direction.csv`, `bootstrap.csv` | secondaries S1 to S3 |
| `tests.csv` | 3 primary and 12 secondary tests in the ledger's format |
| `run_info.json` | plan commit, input hashes, versions, run time (148 seconds) |

```
ln -s <path to raw data> data/raw          # files must match data/MANIFEST.csv
PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 python experiments/p113_connectedness_settings.py   # one process, about 2.5 minutes
PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 python -m pytest -q tests/test_p113_connectedness_settings.py
```

The tests use SYNTHETIC data only: they check the unnormalised decomposition of a known VAR(1) against a
hand calculation, the Cholesky shares against statsmodels, the batched decompositions against the
package functions, the rolling fits against direct fits, the segmented and concatenated designs, the
BIC lag choice, the ranking tie-breaks, and that the shuffle guard and the decision rule recognise a
planted fit and reject a random grid.
