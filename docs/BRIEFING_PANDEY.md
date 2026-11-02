# Briefing: Prof. Pandey project (Nifty expiry and dealer gamma)

*For Shreyan. Updated 3 November 2026 (about 05:30 IST). Plain-language summary of where the project stands. **Everything below is from real NSE data** (2 January 2023 to 2 November 2026: 942 trading days, 200 Nifty expiry days) unless it says "simulation". Each point names the document with the full detail.*

## The one-paragraph version

We downloaded four years of real NSE data, rebuilt the professor's paper on it, and then tested whether option dealers' hedging (measured as "GEX", dealer gamma exposure) explains expiry-day volatility. The paper's "after" period replicates almost exactly, and we think we found its "before" period (14 November 2022 to 10 February 2023, not the dates printed). A hybrid model beats the paper's model for short-horizon forecasts. Dealer gamma does not help forecasts, and most in-sample GEX links fall apart under hard testing, with one exception: Nifty reverses into the close, and only after evenings when the "Pro" traders' positions imply negative GEX. That result survived every attempt to break it. What we cannot pin down with public data is who the dealers are, which is exactly where the professor's knowledge helps.

## Findings, in order of how much to trust them

1. **The data are right.** Minute-bar highs and lows match NSE's official values within 0.05% on 943 of 943 days (highs) and 942 of 943 (lows). Option open interest adds up exactly to NSE's participant totals. ([REAL_DATA_RESULTS.md](REAL_DATA_RESULTS.md), R0)
2. **The paper replicates, with one question for the authors.** Same 21,239 Nifty returns, same extremes, nearly the same model parameters for the "after" period. The printed "before" dates (November 2023 to February 2024) do not match; a search over every 64-day window found 14 November 2022 to 10 February 2023, which matches the printed maxima to every decimal. ([REPLICATION_NSE.md](REPLICATION_NSE.md))
3. **The paper's model can be beaten at every forecast horizon** by about 6% to 8% (a hybrid that keeps the paper's model and corrects its errors with machine learning): 5.8% for the next 15 minutes, 7.6% for the next 30 minutes and 6.4% for the last hour of the day. This is from a second, stronger contest in which the paper's model is refitted as often as ours, and it also holds on data the design never saw. The last-hour edge is the shakiest (the hybrid wins on only 54% of days). We first claimed a simpler model (HAR) was 11% better for the last hour; that came mostly from NSE's new closing auction (from 3 August 2026) and was withdrawn. (R1; [MODEL_ITERATIONS_REAL.md](MODEL_ITERATIONS_REAL.md))
4. **Expiry days are more volatile than the rest of their week, but it depends on the rules.** About 11% more before November 2024, no clear difference from November 2024 to August 2025, about 21% more since Nifty moved to Tuesday, mostly in the last hour. When Bank Nifty's expiry moved to Wednesday, its end-of-day premium moved with it. (R2; [EVENT_STUDY.md](EVENT_STUDY.md))
5. **Pinning (the index closing near a strike on expiry) happened under the Thursday weekly** (4.5 points closer than on other days, p = 0.0006) but not since the move to Tuesday, and it is unrelated to dealer gamma. A test written down in advance found that none of six market-structure explanations explains when it comes and goes. ([PINNING_STUDY.md](PINNING_STUDY.md), [PINNING_MECHANISM.md](PINNING_MECHANISM.md))
6. **The robust GEX result: end-of-day reversal.** A 1% move over the day tends to be followed by about a 0.03% move back in the last 30 minutes (small, but systematic), and this happens only after evenings when GEX under the Pro rule was negative. It survived shuffles, placebo dates, random sign rules, an out-of-sample test fixed in advance and a correction for 78 tests. Caveat: it is driven by day-to-day changes in the Pro traders' positions, not by the option chain itself. It is not a money-making signal: a trading rule written down before looking at the data earns about 4 basis points per trade before costs and loses about 1 after them. ([FALSIFICATION.md](FALSIFICATION.md), [F2_ECONOMICS.md](F2_ECONOMICS.md))
7. **The GEX results that did not survive.** "Higher GEX, calmer next day" turned out to track the total amount of option gamma, not who holds it. The "sign flip on expiry day" holds only in sample, only under the Pro rule, makes forecasts worse and does not appear in Bank Nifty. Expiry-day gamma re-priced minute by minute shows no effect. ([FALSIFICATION.md](FALSIFICATION.md); [INTRADAY_GAMMA.md](INTRADAY_GAMMA.md))
8. **We could not identify the dealer sign.** Letting the data choose the sign gives nearly opposite answers for the volatility outcome and the end-of-day outcome, and no estimated sign forecasts better than the fixed guesses. This is a real, documented negative result, and the best question to ask the professor. ([DEALER_SIGN.md](DEALER_SIGN.md))

**How much to trust each finding.** We ran 4,930 statistical tests in total. [TEST_LEDGER.md](TEST_LEDGER.md) lists every one, with a p-value corrected for all of them at once. The forecasting result survives; the end-of-day reversal survives in sample; the expiry-day sign flip does not.

## What went wrong and got fixed (the honest part)

* An independent reviewer found two overstated claims (GEX "holds under all three sign rules"; a misleading loss comparison). Both were corrected. ([INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md))
* NSE's closing auction from 3 August 2026 freezes the index from 15:16 to 15:28. It inflated one headline result, which was corrected.
* Bugs found and fixed with tests: unsorted strikes in NSE's new file format, a holiday-shifted expiry in June 2023, two edge cases in the volatility model, a stale VIX lag and a missing-value coding in the end-of-day study (numbers changed only in the fourth decimal).

Full log of all 33 iterations: [ITERATION_LOG.md](../ITERATION_LOG.md).

## Decisions taken

* **Every result is reported under three sign rules** (US convention, NSE "Pro" positions, mirror of retail "Client" positions), because nobody can see dealers' positions directly.
* **Out-of-sample forecasting is the headline test,** in-sample regressions are secondary, and every in-sample result went through falsification before being called a finding.
* **Negative results are reported as results.** Most GEX links failed; saying so is what makes the one that survived believable.

## Still running or next (as of 3 November, early morning UTC)

* Done: a stronger forecasting study (option-implied features, the paper's model refitted as often as ours, a hold-out period looked at only once). The hybrid now wins at all three horizons, including the last hour ([MODEL_ITERATIONS_REAL.md](MODEL_ITERATIONS_REAL.md)). Every study now reads this run's data, and the whole project is being regenerated from the raw downloads with one command to check that it reproduces.
* Done: a power study calibrated on the real data ([POWER_REAL.md](POWER_REAL.md)). Our tests could only see effects about as large as the ones we found; effects half that size would need 5 to 11 more years of data. It also showed that the "+11% before November 2024" test is too optimistic, so that result is weak.
* Done: connectedness by gamma regime, a new question for the paper's connectedness design. The pre-stated test is borderline (p = 0.044) and disappears once market volatility (VIX) is controlled ([CONNECTEDNESS_GAMMA.md](CONNECTEDNESS_GAMMA.md)).
* Done: BSE's Sensex expiry as a hidden confounder. BSE's servers are blocked, so the Sensex dates are reported, not verified. The pre-stated test is not significant. Leaving Sensex expiry days out of the comparison makes the NSE expiry premia slightly larger, so our results were understated, not overstated ([SENSEX_CONFOUND.md](SENSEX_CONFOUND.md)).
* Done since the first draft of this briefing: a second person re-typed all 223 numbers we took from the paper, blind; 222 matched and the one difference is a slip in the paper itself ([REKEY_CHECK.md](REKEY_CHECK.md)). The final-half-hour gamma test came out null.
* A forecast for Tuesday 3 November's expiry was frozen in the repository before the market opened; it will be scored after the close.
* A working-paper draft by Wednesday 4 November, 9:30 IST.

## What to send Prof. Pandey

[PITCH_PANDEY.md](../PITCH_PANDEY.md): six findings, five careful-reader questions about the paper (starting with the "before" window), what we would ask for (his view on the Pro category; intraday option-chain data), and two possible papers.

## Limitations

* NSE's participant file covers all index options, not just Nifty, and "Pro" includes traders who may not hedge, so the dealer sign is only a proxy.
* End-of-day positions only: we cannot see how positions change during expiry afternoon.
* BSE's servers are not reachable from our cloud setup, so the Sensex expiry dates used as a control are reported, not verified.
* About 100 weekly expiries since November 2024 is enough for 15- and 30-minute forecasts, not for a last-hour-only test.
