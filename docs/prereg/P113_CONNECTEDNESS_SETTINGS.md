# P113 registered plan: which connectedness setting reproduces the paper's Tables 3 and 4?

Written by an AI agent (Claude) on 3 November 2026, for Shreyan Thanki's internship project with
Prof. Piyush Pandey (IIT Bombay). Status: registered plan (phase 1). **This plan is committed and pushed
before any connectedness number of this study is computed.** No connectedness table under any new setting
has been computed on real data. The only real-data step so far is a check that the raw files match
`data/MANIFEST.csv` (5,213 files, none missing or different). Phase 2 writes
`results/p113_connectedness_settings/` and `docs/P113_CONNECTEDNESS_SETTINGS.md`.

## 1. The question, in plain words

The anchor paper (Chhimwal, Pandey and Pandey 2025, Review of Derivatives Research 28(3), "the paper")
measures how strongly the volatility of the Nifty 50 and of Bank Nifty feed into each other
("connectedness", Diebold and Yilmaz 2012). Its Tables 3 and 4 give 15 small 2 x 2 tables: five
sub-samples (all expiry days, monthly expiry days, and the first, middle and last trading hour of monthly
expiry days) before the change of Bank Nifty's expiry day, and the same five after it, once around Nifty
expiry days (Panel I) and once around Bank Nifty expiry days (Panel II).

Each 2 x 2 table has two "cross shares": the part of Nifty's forecast error variance that comes from Bank
Nifty, and the reverse. The paper's "total" is the sum of the two cross shares (the standard
Diebold-Yilmaz index is half of it).

The project's replication (`docs/REPLICATION_NSE.md`, Tables 3 and 4) reproduces the overall fall in
connectedness but not the one-hour numbers. Three printed totals lie outside every one of the 40 settings
it tried:

| Cell | Printed total (paper) | Range over the replication's 40 settings (REAL DATA, inferred window) |
|---|---|---|
| Before, first hour of monthly expiry days | 99.42 | 55.3 to 98.9 |
| Before, last hour of monthly expiry days | 105.22 | 44.1 to 86.5 |
| After, Nifty expiry, first hour | 80.18 | 54.3 to 73.7 |

(Ranges from `results/replication_nse/tables34_grid_summary.csv`.) A total above 100 needs both cross
shares above 50%: each index would have to get more of its forecast error variance from the other index
than from itself. With the normalised generalized decomposition that the replication uses, this almost
never happens (the largest cross share in the replication's grid for the before last hour is 51.5%,
REAL DATA, same file).

**P113 asks: which standard specification, if any, reproduces the paper's 15 printed totals?** The useful
outcome is either a precise question the authors can answer ("did you normalise the variance
decomposition?") or a clear statement that no standard setting reproduces the hour-level numbers.

## 2. What the paper says about its method (read before writing this plan)

From Sections 3 and 3.2 and the table notes (pages 5 to 11 of the PDF; the PDF is not in the repository):

* **Decomposition.** Section 3.2 names Diebold and Yilmaz (2012) and writes the generalized forecast error
  variance decomposition (its equation 12, with the sum over horizons running from f = 0 to F), and then a
  normalisation (its equation 13) that divides each share by a sum of shares. In the paper's words, just
  before equation 13: "Therefore, the total of the columns does not equal one by definition in the variance
  decomposition matrix". Diebold and Yilmaz (2012) normalise each **row**. The paper's text says
  **columns**, and the summation index in equation 13 is garbled, so the formula can be read either way.
  The printed tables themselves have rows that sum to 100 (except Table 4 Panel I, first hour, whose rows
  sum to 101), which fits row normalisation, or column normalisation printed transposed.
* **Input series.** Section 3 says that the paper first captures the conditional variance of the intraday
  returns with mcsGARCH and then applies the variance decomposition. So the paper's own words point to
  the **mcsGARCH conditional variance** as the input. The title and abstract speak of volatility. The
  replication used the conditional volatility (the square root).
* **Package, VAR lag and horizon.** The paper names **no software package**, **no VAR lag** (it writes "m
  lags") and **no horizon** (it writes "F"). The only lag rule it mentions is in the note to Table 2, for the
  ARMA-mcsGARCH model: lags chosen by the Akaike information criterion (AIC) and the log-likelihood.
* **Sample construction.** The paper does not say whether the expiry days (or hours) were joined end to end
  before fitting the VAR. Section 4.2 is headed "Subsample analysis (dynamic analysis)", and Figure 1 shows
  connectedness over time, which suggests a rolling-window analysis was also run.

So the **main candidate** is the one the paper writes down: the generalized decomposition, normalised
(row-wise as in Diebold and Yilmaz 2012), on the mcsGARCH conditional variance, with the lag and horizon
unknown. The other candidates below are departures from the written method, each included for a stated
reason.

## 3. What was looked at before writing this plan

* The paper's PDF (text layer of all pages; page 8, the connectedness method, also as an image).
* `docs/REPLICATION_NSE.md`, `docs/P108_BLIND_WINDOW.md`, `docs/prereg/P108_BLIND_WINDOW.md`,
  `docs/REKEY_CHECK.md`, `src/niftygex/connectedness.py`, `src/niftygex/replication.py` (VAR and
  connectedness helpers), `experiments/replicate_paper.py`.
* Committed results: `results/replication_nse/tables34_connectedness.csv`, `tables34_connectedness_grid.csv`,
  `tables34_grid_summary.csv`, `rekey_blind.csv`, `expiry_calendar.csv`, `fingerprint_search.csv`.
  From the committed files (REAL DATA), the replication's main setting fits the 15 printed totals with a
  mean absolute difference of 9.26 points (2 of 15 within 2 points), and the best of its 40 settings
  (conditional volatility, AIC lag, horizon 30) with 7.58.
* No connectedness was computed under any setting before this plan was committed.

## 4. Targets (the printed numbers)

Typed from **`results/replication_nse/rekey_blind.csv`** (the blind re-key of the PDF, rows for Table 3,
page 10, and Table 4, page 11), and checked in code against the `P_T34` dict in
`experiments/replicate_paper.py` (the two agree on all 75 cells, `docs/REKEY_CHECK.md`).

| Cell | Sub-sample | Printed total | Printed cross shares (Nifty from Bank Nifty, Bank Nifty from Nifty) |
|---|---|---|---|
| T3.1 | before, all expiry days | 84.00 | 42.71, 41.29 |
| T3.2 | before, monthly expiry days | 74.70 | 36.73, 37.97 |
| T3.3 | before, first hour, monthly | **99.42** | 49.67, 49.75 |
| T3.4 | before, middle hour, monthly | 68.64 | 33.74, 34.90 |
| T3.5 | before, last hour, monthly | **105.22** | 52.79, 52.43 |
| T4.I.1 | after, Nifty expiry days | 76.38 | 38.43, 37.95 |
| T4.I.2 | after, Nifty monthly expiry days | 74.58 | 36.79, 37.79 |
| T4.I.3 | after, Nifty monthly, first hour | **80.18** | 40.14, 40.03 |
| T4.I.4 | after, Nifty monthly, middle hour | 68.81 | 33.68, 35.13 |
| T4.I.5 | after, Nifty monthly, last hour | 66.44 | 29.17, 37.27 |
| T4.II.1 | after, Bank Nifty expiry days | 73.02 | 37.45, 35.57 |
| T4.II.2 | after, Bank Nifty monthly expiry days | 72.74 | 37.13, 35.61 |
| T4.II.3 | after, Bank Nifty monthly, first hour | 70.97 | 37.56, 34.41 |
| T4.II.4 | after, Bank Nifty monthly, middle hour | 66.54 | 33.97, 32.57 |
| T4.II.5 | after, Bank Nifty monthly, last hour | 74.43 | 38.81, 35.62 |

The three **outlying cells** are T3.3, T3.5 and T4.I.3. All 15 totals are the primary targets; the 30
cross shares are secondary targets.

Two cells contain printed slips (`docs/REKEY_CHECK.md`): in T4.I.3 both rows sum to 101, and in T4.II.3 the
cross shares sum to 71.97 while the total is printed as 70.97. The printed totals are the primary targets.
A pre-declared sensitivity analysis (S5) uses **78.17** for T4.I.3 (the total implied if the printed own
shares, 60.86 and 60.97, are right) and **71.97** for T4.II.3 (the sum of the printed cross shares).

## 5. Data and samples (all REAL DATA)

Everything is built exactly as in `experiments/replicate_paper.py` (functions imported from it, not
copied): third-party 1-minute bars, 360 bars a day (09:15 to 15:14), log returns run through from one day to
the next, the expiry calendar from NSE's bhavcopy (`results/replication_nse/expiry_calendar.csv`), monthly
expiry = futures expiry day, first hour 09:15 to 10:14, middle hour 11:45 to 12:44, last hour 14:15 to
15:14. mcsGARCH (ARMA(1,1), Student t, daily variance from an AR(1)-EGARCH(1,1)-t) is fitted separately to
each index in each window, as in the replication.

Before windows:

* **A (primary): 14 November 2022 to 10 February 2023**, the replication's inferred window.
* **B (secondary): 22 November 2022 to 20 February 2023**, P108's blind answer. A and B share 58 of 64
  days and the same three monthly expiry days (24 November 2022, 29 December 2022, 25 January 2023), so the
  four monthly cells differ between A and B only through the mcsGARCH fit.
* **ST (secondary): 1 November 2023 to 29 February 2024**, the window the paper states (82 trading days).

After window: 1 March to 31 May 2024 for all three (as in the replication; Panel I uses Nifty expiry days,
Panel II Bank Nifty expiry days). Table 3 cells use the days on which either index expired, as in the
replication.

## 6. The grid of specifications

Every combination of the following. The main candidate (Section 2) is marked with a star.

**D. Decomposition (5 values)**

| Code | What | Why it is in the grid |
|---|---|---|
| GR* | Generalized decomposition (Pesaran and Shin 1998), each row divided by its sum | The paper's equations 12 and 13, read as in Diebold and Yilmaz (2012); the replication's choice |
| GU | Generalized decomposition, not normalised; total = sum of the two raw cross shares | Raw shares can sum to more than 100% per row, the only simple way to get both cross shares above 50 with symmetric shares; `gfevd(..., normalise=False)` |
| GC | Generalized decomposition, each column divided by its sum | The paper's text says "columns"; printed transposed it would also give rows that sum to 100 (the total is the same either way; the two cross shares swap labels) |
| CN | Cholesky (orthogonalised) decomposition, Nifty ordered first | The decomposition of Diebold and Yilmaz (2009) and of many packages' default variance decomposition |
| CB | Cholesky, Bank Nifty ordered first | The same, other order |

**I. Input series (8 values)**, each a minute-by-minute series for both indices:

| Code | Series | Why |
|---|---|---|
| cvar* | mcsGARCH conditional variance (daily x diurnal x intraday part) | What the paper's Section 3 says it captures before the decomposition |
| cvol | its square root, the conditional volatility | The replication's main input |
| logvar | log of the conditional variance | Log volatility is a common input in connectedness work (any scaling of log volatility gives the same shares) |
| qvol | the mcsGARCH intraday (stochastic) part only, square root | The part of mcsGARCH that is not daily level or time of day |
| ret | 1-minute log returns | Return spillovers; the replication's closest setting for T3.1 |
| absret | absolute 1-minute returns | A simple volatility proxy |
| sqret | squared 1-minute returns | A simple variance proxy |
| rsvol | Rogers-Satchell volatility of each 1-minute bar (square root of the range-based variance) | The paper's own volatility measure in Table 6 |

Not included: realized variance per hour (it gives 3 or 4 numbers per one-hour sub-sample, too few for a
VAR) and range-based measures other than Rogers-Satchell (the paper uses Rogers-Satchell).

**H. Forecast horizon (8 values)**, as the number of steps h = 0, ..., H - 1 summed in the decomposition:
1, 2, 5, 10, 11, 20, 50, 100. H = 1 uses only the contemporaneous correlation; 10 is the horizon of
Diebold and Yilmaz (2012); 11 is the paper's equation 12 with F = 10 (it sums f = 0 to F); 50 and 100 are
long horizons at which the shares have nearly settled.

**C x L. Sample construction and VAR lag (17 pairs)**

* **S, segmented** (the replication's choice): VAR lags never cross from one day, or one hour block, to the
  next. Lags 1, 2, 3, 5, 10, AIC-chosen and BIC-chosen (maximum 10, all orders compared on the same rows):
  7 pairs.
* **C, concatenated**: the sub-sample's days (or hour blocks) are joined end to end into one series and
  the VAR runs straight through, so the first minute of one block is regressed on the last minutes of the
  previous block. This is what a researcher gets by extracting the sub-sample rows and fitting a VAR. Same 7
  lags.
* **R, rolling average**: a VAR is fitted in a rolling window of 200 minutes over the whole window's joined
  series (all days); each sub-sample's table is the average of the rolling tables whose window ends on a
  minute inside that sub-sample. Reason: Section 4.2 is headed "dynamic analysis" and Figure 1 is
  time-varying, so the tables might be averages of rolling tables. 200 is the rolling window length of
  Diebold and Yilmaz (2012) (in days there). Lags 1, 2 and 5 only (3 pairs): choosing a lag by AIC or BIC
  in each of about 30,000 windows is not a standard practice and is left out.

**Size of the grid:** 5 x 8 x 8 x 17 = **5,440 specifications** per before window, each giving 15 totals
and 30 cross shares. Over the three before windows 16,320 (the 10 after cells are shared). Some
specifications give identical or nearly identical numbers (for example H = 10 and H = 11); the number of
distinct ones (15 totals equal after rounding to 0.01) will be reported.

Held fixed (not searched): the session rule, the hour definitions, the expiry calendar, the mcsGARCH
specification, a constant in every VAR, and the first minute of each day (kept, with the overnight gap).

**Baseline check.** Specification (GR, cvol, H = 10, segmented, lag 2) is the replication's main setting.
It must reproduce `results/replication_nse/tables34_connectedness.csv` exactly (all cells, windows A and ST
and after; differences above 1e-9 are a bug that is fixed before anything else).

## 7. Fit criterion and ranking (fixed now)

For each specification, on the primary window A, with e_c = (our total - printed total) for cell c:

* **MAD15 (primary):** the mean of |e_c| over the 15 cells, in points of the paper's scale.
* **K2:** the number of cells with |e_c| <= 2.
* **OUT3:** the largest |e_c| over the three outlying cells.
* **MAD10a:** the mean of |e_c| over the 10 after cells. Reason: our after sample matches the paper's
  Table 1 to the printed digits (21,239 Nifty returns; the same extremes), so for the after cells the
  right setting should come close to the printed numbers, while the before cells also carry the window and
  data-feed uncertainty.
* **CS30 (secondary):** the mean absolute difference over the 30 printed cross shares.

**Ranking:** by MAD15 (lower is better), then K2 (higher), then OUT3 (lower), then a fixed order that
puts the paper's written method first (D: GR, GU, GC, CN, CB; I: cvar, cvol, logvar, qvol, ret, absret,
sqret, rsvol; C: S, C, R; then lower lag, then lower H). The **best specification** is the first in this
ranking.

**Near-ties:** every specification with MAD15 within 0.5 points of the best is listed in
`near_ties.csv` and summarised by how many of them share each value of each dimension. If the near-tie
set mixes decompositions or inputs, the write-up says that the data cannot choose between them.

**Subsets reported separately** (same criterion): (i) the paper's written method, GR with cvar or cvol,
constructions S and C; (ii) all GR specifications ("standard Diebold-Yilmaz 2012"); (iii) each
decomposition; (iv) each construction.

## 8. Guards against fitting a large grid to 15 numbers

With 5,440 settings, some setting will fit well by luck. Two guards, both fixed now:

1. **Shuffle (permutation) reference.** The 15 printed totals are randomly reassigned to the 15 cells
   1,000 times (numpy `default_rng(113)`). For each shuffle the same grid is searched again. For any claim
   of the form "some setting achieves criterion X", the permutation p-value is
   (1 + number of shuffles in which some setting achieves X on the shuffled targets) / 1,001. A claim
   passes only if p <= 0.05. In plain words: if the grid can fit the same numbers just as well when they
   are attached to the wrong cells, a good fit is not evidence for the setting. The same is done for the
   after-only criterion, shuffling the 10 after targets among the 10 after cells.
2. **Leave one table out.** The 15 cells form three tables: Table 3, Table 4 Panel I and Table 4 Panel II.
   For each, the best specification is chosen on the other 10 cells only, and its error on the 5 held-out
   cells is recorded. The **held-out MAD** is the mean of the 15 held-out errors. It is compared with the
   best in-sample MAD15 and with the replication's main setting (9.26 on window A, REAL DATA, Section 3).

## 9. Decision rule (primary window A, whole grid)

* **Reproduced** if the best specification has MAD15 <= 2.0, K2 >= 12, all three outlying cells within 2
  points, held-out MAD <= 3.0, and permutation p <= 0.05 for its MAD15.
* **Partly reproduced** if not reproduced and at least one of:
  * (a) the best specification has MAD15 <= 4.0 and K2 >= 8, held-out MAD <= 5.0, and permutation
    p <= 0.05 for its MAD15;
  * (b) some specification brings all three outlying cells within 5 points while its MAD15 <= 5.0, and the
    permutation p for this criterion (the three cells that receive the values 99.42, 105.22 and 80.18
    within 5 points, MAD15 <= 5.0) is <= 0.05;
  * (c) the after panel is reproduced: some specification has MAD10a <= 1.0 with all 10 after cells within
    2 points, and the permutation p for this criterion (10 after targets shuffled) is <= 0.05.
* **Not reproduced** otherwise.

The same rule is also applied, as secondary results, to window B, to the stated window ST, to the grid with
the window as a sixth dimension (16,320 specifications), and to the subsets of Section 7. The registered
verdict is the one for window A.

**What would be reported to the authors.** If one decomposition or construction stands out (for example
GU), the question for the authors names it. If nothing reproduces the outlying cells, the write-up says
so plainly, with the largest totals each decomposition can produce in those cells.

## 10. Secondary analyses (fixed now)

Under the best specification on window A, and also under the best specification within the paper's
written method (subset (i)) if that is a different specification:

* **S1, changes.** The 10 changes after minus before (Table 4 Panel I and Panel II minus Table 3, for each
  sub-sample), against the printed changes: mean absolute difference, number within 2 points, number with
  the same sign.
* **S2, direction of spillover.** Net spillover of Nifty (to others minus from others; on a 2 x 2 table,
  Bank Nifty from Nifty minus Nifty from Bank Nifty) in each of the 15 cells: number of cells whose sign
  agrees with the printed table. Also over all days of each window: is Nifty a net receiver before, and
  more so after (the paper's Section 4.1)?
* **S3, the paper's conclusions, with day-level bootstrap intervals.** 999 bootstrap draws per cell. A draw
  resamples whole days (whole hour blocks for the hour cells) with replacement: segmented specifications
  keep each drawn block separate, concatenated ones join the drawn blocks in the drawn order, rolling ones
  average the stored rolling tables of the drawn days. AIC or BIC lags are chosen again in each draw. The
  mcsGARCH series are not refitted in the draws. Six statements, each with a 95% percentile interval and a
  two-sided bootstrap p-value, p = min(1, 2 x min((1 + draws <= 0) / 1,000, (1 + draws >= 0) / 1,000)),
  Holm-adjusted within the six for each specification:
  1. All expiry days: Panel I total minus Table 3 total (connectedness falls on Nifty expiry days).
  2. All expiry days: Panel II total minus Table 3 total.
  3. Last hour: Panel I minus Table 3.
  4. Last hour: Panel II minus Table 3.
  5. Last hour change minus all-expiry change, Panel I (the fall is larger in the last hour).
  6. The same for Panel II.
  Before and after draws are independent (different days). In statements 5 and 6 the last-hour and
  all-expiry draws of the same period are also drawn independently, which ignores that they share days and
  makes those intervals wider (conservative).
* **S4, windows.** The decision rule on windows B and ST, and on the grid with the window as a dimension.
* **S5, printed slips.** The best specification and its MAD15 with the alternative targets 78.17 (T4.I.3)
  and 71.97 (T4.II.3).
* **S6, reach of each decomposition.** For each decomposition and each outlying cell: the largest total
  and the largest smaller-of-the-two cross shares over all its specifications, and the number of its
  specifications within 2 and within 5 points of the printed total. This answers directly whether a total
  above 100 is possible at all under each decomposition on our data.

## 11. Hypothesis tests and `tests.csv`

`results/p113_connectedness_settings/tests.csv`, with the columns `test_id, hypothesis, role, sample,
estimate, statistic, p_raw, sidedness, p_method, p_adjusted, adjust_method`:

* **Primary (3 rows):** the permutation p-values of Section 8 for (R / a) the best MAD15, (b) the
  outlying-cell criterion, (c) the after-only criterion. One-sided (a fit at least as good on shuffled
  targets). Holm over the 3 rows.
* **Secondary (6 rows per reported specification):** the S3 bootstrap p-values, two-sided, Holm within the
  six rows of each specification.

## 12. What would change the plan

Nothing is changed after real-data numbers are seen without being reported as a deviation in the
write-up. No setting outside this grid is searched. If a specification cannot be computed (for example a
VAR that cannot be fitted on a cell), it is reported with missing values and ranked last; the count of
such cases is reported.

## 13. Files

* `experiments/p113_connectedness_settings.py`: the run (imports the replication's data and mcsGARCH code).
  Run with `PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 python experiments/p113_connectedness_settings.py`
  (one process).
* `src/niftygex/connectedness.py`: new functions for the column-normalised and Cholesky decompositions;
  existing functions and defaults unchanged.
* `tests/test_p113_connectedness_settings.py`: SYNTHETIC tests only (hand-calculated decompositions of a
  known VAR, rolling fits against direct fits, the shuffle guard, the segmented and concatenated designs).
* Outputs in `results/p113_connectedness_settings/`: `targets.csv`, `baseline_check.csv`,
  `grid_cells.csv.gz` (every specification and cell), `spec_fit.csv` (every specification's MAD15, K2,
  OUT3, MAD10a, CS30, per window), `best.csv`, `near_ties.csv`, `by_dimension.csv`, `subsets.csv`,
  `loto.csv`, `permutation.csv`, `reach.csv`, `changes.csv`, `direction.csv`, `bootstrap.csv`,
  `windows.csv`, `alt_targets.csv`, `verdict.csv`, `tests.csv`, `run_info.json` (plan commit, input
  hashes, versions, run time).
* Write-up: `docs/P113_CONNECTEDNESS_SETTINGS.md`.
