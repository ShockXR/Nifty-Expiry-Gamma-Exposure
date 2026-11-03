# Briefing: Prof. Pandey project (Nifty expiry and dealer gamma)

*For Shreyan. Updated 3 November 2026 (about 05:30 IST). Plain-language summary of where the project stands. **Everything below is from real NSE data** (2 January 2023 to 2 November 2026: 942 trading days, 200 Nifty expiry days) unless it says "simulation". Each point names the document with the full detail.*

## The one-paragraph version

We downloaded four years of real NSE data, rebuilt the professor's paper on it, and then tested whether option dealers' hedging (measured as "GEX", dealer gamma exposure) explains expiry-day volatility. The paper's "after" period replicates almost exactly, and we think we found its "before" period (14 November 2022 to 10 February 2023, not the dates printed). A hybrid model beats the paper's model for short-horizon forecasts. Dealer gamma does not help forecasts, and most in-sample GEX links fall apart under hard testing, with one exception: Nifty reverses into the close, and only after evenings when the "Pro" traders' positions imply negative GEX. That result survived most attempts to break it, but it reflects the Pro traders' positions rather than option gamma, and it is too small to trade. What we cannot pin down with public data is who the dealers are, which is exactly where the professor's knowledge helps.

## Findings, in order of how much to trust them

1. **The data are right.** Minute-bar highs and lows match NSE's official values within 0.05% on 943 of 943 days (highs) and 942 of 943 (lows). Option open interest adds up exactly to NSE's participant totals. ([REAL_DATA_RESULTS.md](REAL_DATA_RESULTS.md), R0)
2. **The paper replicates, with one question for the authors.** Same 21,239 Nifty returns, same extremes, nearly the same model parameters for the "after" period. The printed "before" dates (November 2023 to February 2024) do not match; a search over every 64-day window found 14 November 2022 to 10 February 2023, which matches the printed maxima to every decimal. ([REPLICATION_NSE.md](REPLICATION_NSE.md))
3. **The paper's model can be beaten for the next 15 and 30 minutes, even when it is given the same information** (5.8% and 7.6% better; still 4.8% and 5.9% against a version of the paper's model that also sees India VIX or option prices). The winner is a hybrid that keeps the paper's model and corrects its errors with machine learning. For the last hour the 6.4% gain is mostly the information in VIX, not the method: the paper's model given VIX comes within 1.7%. Plain machine learning without the hybrid structure does about as well. These checks were made after an AI referee report, and they also allow for our having tried 31 models. We first claimed a simpler model (HAR) was 11% better for the last hour; that came mostly from NSE's new closing auction (from 3 August 2026) and was withdrawn. (R1; [MODEL_ITERATIONS_REAL.md](MODEL_ITERATIONS_REAL.md), [REFEREE_RESPONSE_FORECAST.md](REFEREE_RESPONSE_FORECAST.md))
4. **Expiry days are more volatile than the rest of their week, but it depends on the rules.** About 11% more before November 2024, no clear difference from November 2024 to August 2025, about 21% more since Nifty moved to Tuesday, mostly in the last hour (a stricter check finds the whole-day part near zero, but the last-hour part holds either way). When Bank Nifty's expiry moved to Wednesday, its end-of-day premium moved with it. (R2; [EVENT_STUDY.md](EVENT_STUDY.md))
5. **Pinning (the index closing near a strike on expiry) happened under the Thursday weekly** (4.5 points closer than on other days, p = 0.0006) but not since the move to Tuesday, and it is unrelated to dealer gamma. A test whose plan was committed before its results were computed (on the same data that showed the pattern) found that none of six market-structure explanations explains when it comes and goes. ([PINNING_STUDY.md](PINNING_STUDY.md), [PINNING_MECHANISM.md](PINNING_MECHANISM.md))
6. **The robust GEX result: end-of-day reversal.** A 1% move over the day tends to be followed by about a 0.03% move back in the last 30 minutes (small, but systematic), and this happens only after evenings when GEX under the Pro rule was negative. It survived shuffles, placebo dates, random sign rules, a split-sample test and a correction for the 78 headline tests, but not a correction for all 742 tests in that study, and the evenings that drive it come in long stretches (95% of 2023 evenings, about one in five from 2025), so there is less independent evidence than the day count suggests: with tests built for that, it holds over the whole sample and from 2025, but not inside 2023 to 2024. The simplest hedging story (dealers short all options, so the close continues the day's move) fails outright: the close reverses instead. ([REFEREE_RESPONSE_F2.md](REFEREE_RESPONSE_F2.md)) Caveat: it is driven by the Pro traders' positions, not by the option chain itself. The same pattern appears if we simply ask whether Pro traders were net short index options the evening before (this agrees with the GEX sign on 916 of 932 evenings), so the "gamma" part adds nothing. It is not a money-making signal: a trading rule whose plan was committed before its results were computed earns about 4 basis points per trade before costs and about zero after them (it depends on the slippage assumed). ([FALSIFICATION.md](FALSIFICATION.md), [F2_ECONOMICS.md](F2_ECONOMICS.md))
7. **The GEX results that did not survive.** "Higher GEX, calmer next day" turned out to track the total amount of option gamma, not who holds it. The "sign flip on expiry day" holds only in sample, only under the Pro rule, makes forecasts worse and does not appear in Bank Nifty. Expiry-day gamma re-priced minute by minute shows no effect. ([FALSIFICATION.md](FALSIFICATION.md); [INTRADAY_GAMMA.md](INTRADAY_GAMMA.md))
8. **We could not identify the dealer sign.** Letting the data choose the sign, the end-of-day outcome points clearly to the reverse of the Pro positions while the volatility outcome points nowhere clearly (we first wrote that the two point in nearly opposite directions; a check found that is not established), and no estimated sign forecasts better than the fixed guesses. This is a real, documented negative result, and the best question to ask the professor. ([DEALER_SIGN.md](DEALER_SIGN.md))

**How much to trust each finding.** We ran 4,930 statistical tests in total. [TEST_LEDGER.md](TEST_LEDGER.md) lists every one, with a p-value corrected for all of them at once. The forecasting result survives; the end-of-day reversal survives in sample; the expiry-day sign flip does not.

## What went wrong and got fixed (the honest part)

* A separate AI reviewer found two overstated claims (GEX "holds under all three sign rules"; a misleading loss comparison). Both were corrected. ([INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md))
* NSE's closing auction from 3 August 2026 freezes the index from 15:16 to 15:28. It inflated one headline result, which was corrected.
* Bugs found and fixed with tests: unsorted strikes in NSE's new file format, a holiday-shifted expiry in June 2023, two edge cases in the volatility model, a stale VIX lag and a missing-value coding in the end-of-day study (numbers changed only in the fourth decimal).

Full log of all 33 iterations: [ITERATION_LOG.md](../ITERATION_LOG.md).

## Decisions taken

* **Every result is reported under three sign rules** (US convention, NSE "Pro" positions, mirror of retail "Client" positions), because nobody can see dealers' positions directly.
* **Out-of-sample forecasting is the headline test,** in-sample regressions are secondary, and every in-sample result went through falsification before being called a finding.
* **Negative results are reported as results.** Most GEX links failed; saying so is what makes the one that survived believable.

## Still running or next (as of 3 November, early morning UTC)

* Done: a stronger forecasting study (option-implied features, the paper's model refitted as often as ours, a later hold-out period the design choices were not tuned on, although our first run had covered it). The hybrid now wins at all three horizons, including the last hour ([MODEL_ITERATIONS_REAL.md](MODEL_ITERATIONS_REAL.md)). Every study now reads this run's data, and the whole project is being regenerated from the raw downloads with one command to check that it reproduces.
* Done: a power study calibrated on the real data ([POWER_REAL.md](POWER_REAL.md)). Our tests could only see effects about as large as the ones we found; effects half that size would need 5 to 11 more years of data. It also showed that the "+11% before November 2024" test is too optimistic, so that result is weak.
* Done: connectedness by gamma regime, a new question for the paper's connectedness design. The pre-stated test is borderline (p = 0.044) and disappears once market volatility (VIX) is controlled ([CONNECTEDNESS_GAMMA.md](CONNECTEDNESS_GAMMA.md)).
* Done: BSE's Sensex expiry as a hidden confounder. BSE's servers are blocked, so the Sensex dates are reported, not verified. The pre-stated test is not significant. Leaving Sensex expiry days out of the comparison makes the NSE expiry premia slightly larger, so our results were understated, not overstated ([SENSEX_CONFOUND.md](SENSEX_CONFOUND.md)).
* Done since the first draft of this briefing: a separate AI agent re-typed all 223 numbers we took from the paper, blind; 222 matched and the one difference is a slip in the paper itself ([REKEY_CHECK.md](REKEY_CHECK.md)). The final-half-hour gamma test came out null.
* A forecast for Tuesday 3 November's expiry was frozen in the repository before the market opened; it will be scored after the close.
* Done: a forward test written down before the future data exist ([FORWARD_TEST.md](prereg/FORWARD_TEST.md)). The end-of-day reversal and the Tuesday last-hour premium will each be tested once, on new days only, when enough have piled up (about two to three years). That is the only fully honest out-of-sample test available, and the commit time proves the plan came first.
* A working-paper draft by Wednesday 4 November, 9:30 IST.

## What to send Prof. Pandey

[PITCH_PANDEY.md](../PITCH_PANDEY.md): six findings, five careful-reader questions about the paper (starting with the "before" window), what we would ask for (his view on the Pro category; intraday option-chain data), and two possible papers.

## Limitations

* NSE's participant file covers all index options, not just Nifty, and "Pro" includes traders who may not hedge, so the dealer sign is only a proxy.
* End-of-day positions only: we cannot see how positions change during expiry afternoon.
* BSE's servers are not reachable from our cloud setup, so the Sensex expiry dates used as a control are reported, not verified.
* About 100 weekly expiries since November 2024 is enough for 15- and 30-minute forecasts, not for a last-hour-only test.
