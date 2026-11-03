# Blind second transcription of the paper's printed numbers

**Who did it:** a separate AI agent (Claude), working blind from the PDF; not a human. **Result in one line:** a second, blind transcription typed all 223 printed numbers that our replication compares against again, blind, from the PDF. 222 matched the first typing exactly. The one difference is a typesetting slip in the paper itself, and the first typing had already read it correctly. Nothing in the replication changed.

## Why this check was needed

`experiments/replicate_paper.py` recomputes the tables of Chhimwal, Pandey and Pandey (2025, *Review of Derivatives Research* 28, article 14) on real NSE data and puts our numbers next to the paper's. The paper's numbers ("printed values") were typed into the script by hand, in the dicts `P_T1`, `P_T1_CORR`, `P_T2`, `P_T34`, `P_T5`, `P_T5_TEXT_PCT` and `P_T6`.

A typing mistake there (a "transcription slip") would do damage in two ways. It could make the replication look worse than it is. It could also make the paper look inconsistent when it is not. That matters because `docs/REPLICATION_NSE.md` (Section 8) lists a few printed numbers that do not add up, and we want to raise these with the authors. Before that, someone other than the first typist had to check every value against the PDF.

## How it was done

1. **Blind re-key.** Every number the dicts cover was typed again from the PDF into `results/replication_nse/rekey_blind.csv`, before the dict values were opened. Each row gives the table, the page, the row and column labels as printed, the value exactly as printed (as text, with its decimals, sign, thousands commas and significance stars), and a note. Only the dicts' structure (which cells they hold) was looked at beforehand. One disclosure: a search for the dict names printed the start of two one-line dicts, so three values (both `P_T1_CORR` correlations and the first `P_T5_TEXT_PCT` value, 32) were glimpsed before re-keying. All three were still typed from the PDF.
2. **Reading the PDF.** The tables are real text, not pictures, so every number could be read. Pages were viewed as images (at 200 dpi, and at 500 dpi for the one disputed cell). As a check on the second typing itself, every typed table value was also looked up in the PDF's own text layer (all 217 found), and the cell positions of Tables 3 and 4 were checked line by line. The page renders and text were kept in a temporary folder outside the repository and deleted afterwards. The repository holds only individual table numbers and short labels.
3. **Comparison.** `experiments/rekey_check.py` imports the dicts from `replicate_paper.py` (without running the replication), matches every blind row to a dict cell and writes `results/replication_nse/rekey_check.csv`. That file has one row per cell, with both values, `match` (yes or no), the absolute difference, and a `resolution` column for anything that differed. Cells found on only one side are listed too. In this run there were none.
4. **Resolution.** Each mismatch was looked at again in the PDF and decided.

Page numbers: the PDF's page numbers and the article's printed page numbers ("Page 5 of 16") are the same, so one number is given.

## What was checked

| Source | Page | What | Cells | Matched first time |
|---|---|---|---|---|
| Table 1 (descriptive statistics) | 5 | N, mean, SD, min, max, skewness, kurtosis, Anderson-Darling, ADF for 4 columns, plus the 2 correlations | 38 | 37 |
| Table 2 (mcsGARCH) | 7 | AR(1), MA(1), omega, alpha, beta, shape, log-likelihood, AIC for 4 columns | 32 | 32 |
| Table 3 (connectedness before) | 10 | 5 blocks: the four shares and the total | 25 | 25 |
| Table 4 (connectedness after) | 11 | 2 panels x 5 blocks: the four shares and the total | 50 | 50 |
| Table 5 (volume) | 12 | 3 rows x 2 indices: both N, both medians, the statistic, the p-value | 36 | 36 |
| Table 6 (volatility) | 13 | the same layout as Table 5 | 36 | 36 |
| Section 4.3 text | 12 | the six median volume changes quoted in words (32%, 19%, 61%, minus 13%, 9%, minus 3%) | 6 | 6 |
| **Total** | | | **223** | **222** |

Cells the dicts do not hold were not re-keyed: Table 2's constant, z statistics and goodness-of-fit p-values, and the "From" and "To" cells of Tables 3 and 4 other than the total (these just repeat the off-diagonal shares).

## The one mismatch

| Cell | Page, table | Printed | In the dict | Decision |
|---|---|---|---|---|
| `P_T1[after,banknifty].anderson_darling` | page 5, Table 1, After, Nifty Bank Return, Anderson-Darling row | `466,153***` | 466.153 | **Dict is right; the slip is the paper's.** |

The Anderson-Darling statistic measures how far returns are from a normal "bell curve" (bigger means less normal). At 500 dpi, and in the PDF's text layer (character code 0x2C), the paper really prints a comma here. The other three cells of the same row use a decimal point (649.191, 790.061, 492.399). Read literally, with the comma as a thousands separator, the value would be 466,153, 600 to 950 times its neighbours, which is not plausible. Our own statistic for this sample is 469.5 (`table1_descriptive.csv`). So the comma is a typesetting slip in the paper, and the first typist had already read it as 466.153. No constant was changed.

## What changed in the replication

Nothing. No value in `replicate_paper.py` was a transcription slip, so no constant was edited, the replication was not rerun, and no row of `results/replication_nse/*.csv` changed.

Some wording in `experiments/replicate_paper.py` still says the values are "not yet re-keyed by a second person". This is in the module docstring (line 10), the comment above `P_T1`, the `caveat` column it writes into `paper_printed_checks.csv` (line 1031) and one stdout line (line 1124). It was left alone because this check was only allowed to change constants. Updating it needs a rerun, so that the output files carry the new wording.

## The Section 8 "small printed-number points": ours or the paper's?

Every value behind these points was confirmed in the PDF, so **all of them are real inconsistencies in the paper, not transcription slips of ours.**

| Point in `REPLICATION_NSE.md` Section 8 | Where | Values as printed | Verdict |
|---|---|---|---|
| Bank Nifty before: log-likelihood and AIC imply N = 22,680, not 23,039 | page 7, Table 2; page 5, Table 1 | LL 148,500, AIC -13.095; N 23,039. N = 2 x 148,500 / 13.095 = 22,680. The other three columns give about 23,040, 21,241 and 21,240, in line with their N (23,039, 21,239, 21,239). | In the paper |
| Table 4, Panel I, first hour: rows sum to 101 | page 11, Table 4 | 60.86 + 40.14 = 101.00; 40.03 + 60.97 = 101.00 | In the paper |
| Table 4, Panel II, first hour: total should be 71.97 | page 11, Table 4 | 37.56 + 34.41 = 71.97, printed total 70.97 | In the paper |
| Table 5, Nifty on Thursday: medians imply +65%, text says 61% | page 12, Table 5 and Section 4.3 | 469,000 to 774,000 is +65.0%; the text says "increases by 61%" | In the paper |
| Tables 5 and 6: some test statistics cannot match their p-values | pages 12 and 13 | All twelve statistics and p-values are as printed. For example, Table 6, Bank Nifty, Thursday prints 2,174,645,383, while 4,679 x 4,679 = 21,893,041 is the largest a Mann-Whitney U can be. | In the paper |
| New: Table 1 Anderson-Darling printed with a comma | page 5, Table 1 | 466,153 (see above) | In the paper (added to Section 8) |

## Other things noticed while re-keying (no action taken)

* Table 1 prints its "After" negative numbers with a plain hyphen and its "Before" negative numbers with a typographic minus. This is cosmetic and changes no value.
* Table 1's row is labelled "Kurtosis". The dict calls it `excess_kurtosis`, based on the replication's evidence (`REPLICATION_NSE.md`, Section 3). The paper does not say which it is.
* The "after" observation counts differ between tables: N = 21,239 in Table 1, 21,463 in Table 5 and 21,462 in Table 6. These are typed correctly. Why they differ is a question for the authors, not a transcription matter.

## How to rerun

```
python experiments/rekey_check.py                          # compares and writes results/replication_nse/rekey_check.csv
OMP_NUM_THREADS=1 python -m pytest -q tests/test_rekey_check.py   # synthetic test, no PDF needed
```

If a printed value is ever corrected in `replicate_paper.py`, record the first-pass value and the reason in `RESOLUTIONS` in `experiments/rekey_check.py`, so that the check keeps the history.
