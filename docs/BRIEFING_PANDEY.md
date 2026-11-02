# Briefing: Prof. Pandey project (Nifty expiry and dealer gamma)

*Written 2 November 2026 for Shreyan. Plain-language summary of the overnight build. Everything below is from simulated data or from arithmetic on the paper's printed numbers; nothing is from real market data yet.*

## Findings

1. **The paper.** Chhimwal, Pandey and Pandey (2025, Review of Derivatives Research) show that after Bank Nifty's expiry moved to Wednesday, each index got more volatile and busier on its own expiry day, and the two indices moved together less. They fit their volatility model in-sample and do not test why it happened. That "why" is the opening for this project.
2. **The gap is real.** A literature search (35 references checked, none fake) found no peer-reviewed study of option dealers' hedging effects on Nifty or any NSE options.
3. **Careful-reader notes on the paper** (useful to show Prof. Pandey we read it closely, phrased as questions, not criticism):
   * Its connectedness totals are on a scale twice the usual one (84 means 42 in the standard index).
   * Its ARMA(1,1) terms nearly cancel; our simulations show that is what ARMA(1,1) does on data with no real autocorrelation.
   * Three Mann-Whitney statistics are larger than mathematically possible (likely typos), and one pair of significance stars does not match its z statistics.
   * Its numbers imply intraday volatility shocks fade much faster after the change, but our simulations show that estimate is fragile at its sample size.
4. **The paper's Wilcoxon tests overstate significance.** In simulations at its sample sizes where nothing changed, the minute-by-minute test said "significant change" 69% of the time for volatility and 87% for volume, instead of 5%. Testing at the level of days, or comparing each expiry day with the rest of its own week, brings that back to about 2 to 6%.
5. **The gamma test works, with conditions.** In 48 simulated markets, adding dealer-gamma features to a forecast was detected as useful in 62 to 88% of markets when the effect was real and positions were measured well, and almost never when there was no effect. Three conditions matter:
   * The sign rule is decisive: the common US convention never detected the effect in our India-style simulated market.
   * Noisy position data destroy most of the power.
   * Gains are small (1 to 4%) and the last hour of expiry day cannot be tested reliably with end-of-day data alone.

## Decisions taken

* **Headline:** a forecasting contest (does positioning improve out-of-sample forecasts of expiry-day volatility beyond the paper's mcsGARCH), as agreed earlier, rather than an in-sample regression.
* **Build on simulation first,** because this cloud machine cannot reach NSE. The real-data path is built and tested on files in NSE's formats, and runs with one command once data are downloaded.
* **Report every result under several sign rules** (US convention, NSE "Pro" positions, mirror of retail "Client" positions) because the sign cannot be observed directly.
* **Kept out of the main package:** the trading backtest (it would only show profits we built into the simulator), a deep sequence model (not enough expiry days to justify it yet). Both are written up in `alternatives/ALTERNATIVES.md`.

## Validation

* 28 automated tests: greeks against finite differences and put-call parity, GEX against its definition, mcsGARCH parameter recovery, connectedness against a closed form, the expiry calendar against known dates (including the 2 September 2025 switch to Tuesday), and a round trip through NSE file formats.
* mcsGARCH recovers the paper's own parameter values from simulated data; at the paper's sample size its persistence estimate is biased up slightly when the daily part is estimated.
* The implied-vol solver is accurate to $10^{-12}$ for every option worth at least one tick, 50,000 options in under a second.

## Iterations (what went wrong and got fixed)

1. A range-based volatility test exposed a known bias of high-low estimators under coarse sampling; documented.
2. The simulator first published option chains with the wrong timing; fixed so nothing from the future leaks.
3. The daily EGARCH fit diverged on some training windows (forecasts off by 10^8). Added checks and a fallback.
4. The simulator leaked next-day volatility through option implied vols, which made gamma features look useful when they were not. Fixed and the power study rerun.
5. The first Wilcoxon experiment used a null that made every test fail; redesigned into two clean scenarios.

Full log: `ITERATION_LOG.md`.

## Limitations

* No real-market results yet. The simulator's assumptions (including how strong the gamma effect is) are ours.
* NSE participant data cover all index options, not Nifty strikes, so the sign of dealer gamma is only proxied.
* End-of-day GEX only; intraday option-chain history is the main data gap.
* About 100 weekly expiries since November 2024: enough for the 30-minute forecasts, probably not for a last-hour-only test.

## Next actions

1. On your own machine: run `scripts/download_nse.py` for bhavcopy and participant files, and get 1-minute Nifty bars (broker API or vendor). Check OI units on the first file (steps in `docs/DATA_ACQUISITION.md`).
2. Run `python scripts/run_pipeline.py --source nse` and look at the scoreboard.
3. Replicate the paper's Tables 2 to 6 on its window, then redo its volume and volatility tests at day level.
4. Send Prof. Pandey `PITCH_PANDEY.md` with the first real scoreboard, and ask about the "Pro" category and any data his group can share.
