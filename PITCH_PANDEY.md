# Pitch: what positioning adds to your expiry-day result

*For Prof. Piyush Pandey. From Shreyan Thanki, November 2026.*

## The idea in three sentences

Your 2025 paper with Chhimwal and V. Pandey shows that each index became more volatile and more heavily traded on its own expiry day after NSE separated the Nifty and Bank Nifty expiries. I would like to test one candidate explanation, option dealers' gamma hedging, by asking a forecasting question: does knowing dealer positioning help predict expiry-day volatility out of sample, beyond your mcsGARCH model? Your model is the benchmark, so the answer is useful either way.

## Why it is worth doing

* **It turns a documented effect into a tested mechanism.** The paper attributes the expiry-day rise to unwinding, profit booking and retail flows, which the data used could not test. Gamma exposure is measurable from public NSE files (bhavcopy and participant-wise open interest).
* **The gap is real.** I found no peer-reviewed study of dealer gamma effects on Nifty or any NSE options, while US evidence points both ways (Ni, Pearson, Poteshman and White 2021 find an effect; Dim, Eraker and Vilkov 2024 find none for S&P 500 0DTEs).
* **It extends your design to two new natural experiments:** the single weekly expiry rule (20 November 2024) and the move of Nifty weekly expiry to Tuesday (1 September 2025).

## What is already built

A tested Python package (open on GitHub) with a Black-76 greeks engine, GEX under several sign rules, your mcsGARCH rebuilt (with an optional separate intraday shape for expiry days), Diebold-Yilmaz connectedness with bootstrap intervals, a walk-forward forecasting contest (naive, VIX, HAR, mcsGARCH, gradient-boosted trees, and a hybrid that corrects mcsGARCH), and loaders for NSE's file formats. Everything has been validated on simulated markets where the right answer is known. It has not yet been run on real data.

## What I learned from your paper's numbers

Offered as a careful reader's notes, in case they are useful:

1. The connectedness totals in Tables 3 and 4 are the sum of the off-diagonal shares, which is twice the standard Diebold-Yilmaz index for two variables (84.00 corresponds to 42.00). Comparisons within the paper are unaffected.
2. Table 2 implies the half-life of an intraday volatility shock fell sharply after the change (about 339 to 72 minutes for Nifty, 185 to 40 for Bank Nifty). That may be an interesting result in its own right, though half-lives this close to persistence 1 are fragile: in my simulations at your sample size, estimated persistence was biased up by about 0.002, enough to triple a half-life, so it needs standard errors first.
3. In all four columns of Table 2 the AR(1) and MA(1) estimates are nearly equal and opposite, so the fitted mean is close to white noise. Simulations show ARMA(1,1) does this on near-white-noise data.
4. The Wilcoxon tests treat thousands of minutes from 12 or 13 days as independent. In simulations at your sample sizes (64 versus 59 days) where nothing changed, the minute-level test rejected "no change" in 69% of runs for volatility and 87% for volume, against the nominal 5%; day-level tests rejected in about 6%. When daily volatility is persistent, comparing expiry days with the other days of the same week (a difference-in-differences) kept false rejections near 2% and still detected a 25% rise in expiry-day variance about half the time with 13 versus 12 weeks. These are simulation results, not a re-analysis of your data, but they suggest day-level or week-relative versions of the tests would make the findings more robust.
5. A few Mann-Whitney statistics in Table 6 are larger than the maximum possible for their sample sizes, which looks like misplaced digits.

## What I would ask for

* Your view on whether the "Pro" category in NSE participant-wise open interest is a reasonable proxy for dealers.
* Whether your group has intraday option-chain history, or 1-minute Nifty and Bank Nifty data from the 2025 paper that I could use to check my replication of your tables.
* Fifteen minutes to agree on the first deliverable: a scoreboard of out-of-sample forecasts on November 2024 to the present, with the GEX test under each sign rule.

## Timeline

| Week | Deliverable |
| --- | --- |
| 1 | Real data loaded; replication of your Tables 2 to 6 with a sensitivity grid over the unreported VAR settings |
| 2 | Forecast scoreboard and GEX test on November 2024 to the present |
| 3 | Day-level re-tests of the volume and volatility results; the two new natural experiments |
| 4 | Short write-up and, if results support it, a plan for a paper |
