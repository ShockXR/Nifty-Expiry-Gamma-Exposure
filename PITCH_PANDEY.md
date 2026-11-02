# Pitch: what dealer positioning adds to your expiry-day result

*For Prof. Piyush Pandey. From Shreyan Thanki, November 2026. Every result below is from **real NSE data**, 2 January 2023 to 2 November 2026 (942 trading days, 200 Nifty expiries), unless it says otherwise. Each one links to a document in this repository that shows the file it comes from.*

## The idea in three sentences

Your 2025 paper with Chhimwal and V. Pandey shows that each index became more volatile and more heavily traded on its own expiry day after NSE separated the Nifty and Bank Nifty expiries. I tested one candidate explanation, option dealers' gamma hedging, with public NSE files, first by asking whether dealer positioning improves out-of-sample forecasts beyond your mcsGARCH model, and then by trying hard to break every in-sample link I found. The answer is more interesting than a yes or a no, and it gives a concrete plan for a paper.

## What I found

1. **Your "after" sample replicates almost exactly, and I think I found your "before" window.** With 360 minutes a day I get exactly your 21,239 Nifty returns and Table 1's extremes to the printed digit; mcsGARCH gives alpha 0.0498 and beta 0.9407 (printed 0.0494 and 0.9410). The stated "before" period (November 2023 to February 2024) has 82 trading days, not 64, and does not match. Scoring every 64-day window against Table 1 picks out **14 November 2022 to 10 February 2023**, which reproduces both printed "before" maxima to every decimal and agrees on statistics not used in the search. My first question to you is whether that was the sample. ([REPLICATION_NSE.md](docs/REPLICATION_NSE.md))
2. **Your model is a strong benchmark, and it can be improved.** In an out-of-sample contest on 792 test days, a hybrid that keeps mcsGARCH and corrects its errors with a gradient-boosted model forecasts the next 15 and 30 minutes about 7% and 9% better (QLIKE, Diebold-Mariano p of 0.001 or below), still about 6% and 7% better after the opening minute, and better in every expiry regime. For the last hour of the day nothing reliably beats mcsGARCH. ([REAL_DATA_RESULTS.md](docs/REAL_DATA_RESULTS.md), R1)
3. **The expiry effect depends on the expiry regime, and lives at the end of the day.** Compared with the rest of its own week, a Nifty expiry day carried about 11% more variance before November 2024 (weak evidence: a simulation check shows the test over-rejects in that period), no clear excess under the single Thursday weekly (November 2024 to August 2025), and about 21% more since the move to Tuesday, concentrated in the last hour (about 43% more). When Bank Nifty's expiry moved to Wednesday, its last-30-minute premium moved with it (+1.14 log points, placebo p 0.005) while whole-day variance did not. ([REAL_DATA_RESULTS.md](docs/REAL_DATA_RESULTS.md), R2; [EVENT_STUDY.md](docs/EVENT_STUDY.md))
4. **Dealer gamma, measured from public files, does not help forecasts, but one in-sample pattern is unusually robust.** GEX features add nothing out of sample under any of three sign rules. In sample, Nifty reverses into the close (a 100 bp move over the day is followed by about 3 bp the other way in the last 30 minutes), and the reversal appears only after evenings when GEX built from NSE's "Pro" participant positions was negative (slope -0.057 against +0.011 after positive evenings; difference p = 0.0002). That pattern survived every falsification test I could design: block shuffles of GEX, GEX taken from later days, random sign rules (it beats 97% of them, 99.6% with controls), a test on 2025 to 2026 data specified from 2023 to 2024 data (p = 0.006), and a Holm correction over 78 tests. One caveat changes its reading: it is carried by the day-to-day changes in Pro participants' net option positions, not by the shape of the option chain. It is also too small to trade: a rule fixed in advance earns about 4 bp per trade before costs and nothing after them. ([FALSIFICATION.md](docs/FALSIFICATION.md), [F2_ECONOMICS.md](docs/F2_ECONOMICS.md))
5. **The sign of dealer positions is the real open problem.** The robust reversal fits hedging theory only if the Pro positions are the opposite of dealers' positions, while the next-day volatility link fits only if they are the same. When I estimate the sign from the market's response instead of assuming it, the two outcomes point in nearly opposite directions, and no estimated sign beats the fixed rules out of sample. The ordinary-day volatility link also turned out to be a proxy for the total amount of option gamma, not for its sign. ([DEALER_SIGN.md](docs/DEALER_SIGN.md))
6. **Pinning is real but regime-specific.** Under the single Thursday weekly, Nifty's official close on expiry day was 4.5 points closer to a 50-point strike than on other days (p = 0.0006); since the move to Tuesday there is none, and pinning is unrelated to any measure of dealer gamma. A pre-registered test of four market-structure explanations (open interest near the money, volume share, contract size, number of weeklies) found that none explains when pinning comes and goes. ([PINNING_STUDY.md](docs/PINNING_STUDY.md), [PINNING_MECHANISM.md](docs/PINNING_MECHANISM.md))

I also froze a forecast for Tuesday 3 November's expiry in the repository before the market opened, so it can be scored against what actually happened ([results/live/2026-11-03/forecast.md](results/live/2026-11-03/forecast.md)).

## Points from your paper I would like to check with you

Offered as a careful reader's questions, each now checked on real data:

1. **The "before" window** (point 1 above).
2. **Connectedness scale.** Tables 3 and 4 report the sum of the two cross shares, which is twice the standard Diebold-Yilmaz index for two variables (84.00 corresponds to 42.00). Comparisons within the paper are unaffected. The overall fall in connectedness replicates (87.9 to 74.9 and 68.9 on your scale; printed 84.00 to 76.38 and 73.02), but the first- and last-hour totals of 99.42 and 105.22 are outside everything a 40-setting grid of unreported VAR choices produces.
3. **Minute-level tests.** The Mann-Whitney tests in Tables 5 and 6 treat thousands of minutes from a few dozen days as independent. On real data the minute-level test gives p-values as small as 10^-256; tests on whole days keep the Nifty findings (higher volatility overall and on Thursdays) but not the Bank Nifty Wednesday result (+16%, not significant at the day level).
4. **Persistence.** Your Table 2 implies the half-life of an intraday volatility shock fell from about 339 to 72 minutes for Nifty. I find 359 to 72 on the inferred window, with a wide interval on the "before" side (197 to 1,255 minutes), so the fall is clear but its size is uncertain.
5. **A few printed numbers look like typesetting slips** (for example, Table 4's first-hour Nifty rows sum to 101, and one Anderson-Darling statistic in Table 1 is printed with a comma for the decimal point). A second reader re-typed all 223 printed numbers blind from the PDF; 222 matched our first typing and the one difference is that comma, so these points are in the printed paper, not our transcription ([REKEY_CHECK.md](docs/REKEY_CHECK.md)).

## What I would ask for

* **Your view on the "Pro" category** in NSE's participant-wise open interest as a proxy for dealers. My results suggest it is not a clean one, and this is where your knowledge of the Indian market would matter most.
* **Data:** the exact "before" sample dates, and whether your group has intraday option-chain snapshots (positions during the day, not only at the close), which would let me test the hedging story directly on expiry afternoons.
* **Fifteen minutes** to agree which of two directions to write up first.

## Two possible papers

| | A. Dealer positioning and the end-of-day reversal on NSE | B. Expiry rules as natural experiments |
| --- | --- | --- |
| Question | Why does Nifty reverse into the close, and why more after negative Pro GEX? | How do expiry calendars change end-of-day volatility, pinning and connectedness? |
| What exists | The robust reversal result, falsification suite, sign estimation | Your paper's design extended to five rule changes, with comparison indices and placebo dates |
| What is missing | A cleaner dealer proxy (your group's data or a broker category) | Verified BSE Sensex expiry dates (BSE's servers are not reachable from my setup; with reported dates as a control the NSE premia get slightly larger, see [SENSEX_CONFOUND.md](docs/SENSEX_CONFOUND.md)) |
| First deliverable | 2 weeks | 2 weeks |

## How to check any of this

Every one of the project's 4,930 hypothesis tests is listed in [TEST_LEDGER.md](docs/TEST_LEDGER.md) with its raw p-value and a p-value corrected for all of them at once (Benjamini-Yekutieli). The forecasting result survives that correction; the end-of-day reversal survives in sample only.

The code is a tested Python package with one command that downloads the public data (pinned sources), checks every raw file against a list of SHA-256 hashes and reruns every study ([REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)). Each step that went wrong and how it was fixed is logged in [ITERATION_LOG.md](ITERATION_LOG.md), including two of my own headline claims that were withdrawn after an independent review and the closing-auction check ([INDEPENDENT_REVIEW.md](docs/INDEPENDENT_REVIEW.md)).
