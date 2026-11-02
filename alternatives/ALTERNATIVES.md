# Alternative and follow-on directions

These are kept out of the main package on purpose. Each is either blocked on data we do not have yet, or would add claims that synthetic data cannot support. They are listed in the order we would pick them up.

## A1. Trading layer (blocked on real option prices)

**Idea.** Turn the volatility forecast into a decision a desk would recognise, and ask whether it survives real costs.

| Strategy | Signal | What it tests |
| --- | --- | --- |
| Expiry-day short straddle with a filter | Sell the ATM straddle at 09:20 only when forecast variance for the rest of the day is below the variance implied by the straddle price; stand aside when net GEX is negative | Economic value of the forecast |
| Last-30-minute momentum on Nifty futures | Trade with the day's move on negative-GEX days, fade it on positive-GEX days | The hedging-flow mechanism (Baltussen et al. 2021) |
| Pin trade | Iron butterfly at the max-OI strike on positive-GEX expiry days | Pinning (Avellaneda and Lipkin 2003; Golez and Jackwerth 2012) |

**Why it is not in the main package.** A backtest needs intraday option prices. Our simulator produces Black-76 prices with a fixed volatility risk premium, so any profit it shows would be put there by construction. Reporting it would be a fabricated result.

**What it needs.** Intraday Nifty option-chain history (paid vendor) or forward collection from now; the current NSE and SEBI charge schedule (securities transaction tax, exchange transaction charges, SEBI fee, stamp duty, GST), taken from the official circulars on the day of the backtest rather than from memory; the extra 2% expiry-day margin on short options from SEBI's October 2024 circular; a slippage model calibrated on real spreads. Evaluation: net Sharpe, drawdown, turnover, a placebo with shuffled GEX signs, and the deflated Sharpe ratio (Bailey and López de Prado 2014) for the number of variants tried.

## A2. Sequence model (deferred on evidence)

An LSTM or small Transformer on the minute-by-minute history, as the last rung of the forecast ladder. Deferred because (a) there are about 100 weekly expiries since November 2024, which is little independent signal for a sequence model; (b) the daily-frequency Nifty evidence favours GARCH over an LSTM (Mahajan, Thakan and Malik 2022); (c) trees on engineered features plus mcsGARCH already capture the time-of-day and persistence structure. Add it only if the hybrid model clearly beats mcsGARCH on real data, so there is something left for it to learn. The walk-forward engine accepts any model that maps a feature matrix to a log-variance forecast, so it plugs in without changes elsewhere.

## A3. Intraday GEX snapshot collector

End-of-day GEX lagged one day is what free data allow. Intraday option-chain snapshots every 5 minutes would let GEX update during expiry day, when gamma moves fastest. A collector that polls a broker API and appends to Parquet is straightforward; history cannot be backfilled for free, so the value grows with every week it runs.

## A4. Connectedness by gamma regime

Run Diebold-Yilmaz for Nifty and Bank Nifty (the paper's pair) and for Nifty and Sensex (which now expire on different weekdays), split by negative versus positive Nifty GEX days, with block-bootstrap intervals. This ties the paper's connectedness result to hedging. The engine exists (`niftygex.connectedness`); it needs Bank Nifty and Sensex 1-minute bars.

**Done for Nifty and Bank Nifty (RQ4, P42): [docs/CONNECTEDNESS_GAMMA.md](../docs/CONNECTEDNESS_GAMMA.md), `experiments/connectedness_gamma.py`.** The pre-stated test finds +4.29 points of total connectedness after Pro-rule short-gamma evenings (p 0.044). It is not robust: it is zero in a day-level regression with VIX, and no Pro secondary test survives Holm. Only gross gamma (no sign) survives, with lower connectedness on high-gamma days. The Nifty and Sensex pair is still open (it needs Sensex minute bars).

## A5. Momentum regression

$r_{\text{last 30}} = a + b\, r_{\text{open to 15:00}} + c\, r_{\text{open to 15:00}} \cdot \mathbb 1[\text{GEX} < 0] + e$, HAC standard errors, expiry and non-expiry days separately. A positive $c$ is the hedging prediction. Not run on synthetic data because the simulator has no momentum channel, so the answer would be zero by construction.

## A6. Isolating the September 2023 Bank Nifty weekly change

The paper's own stated future work. Same design as the paper, with the day-level tests from `niftygex.inference`, on August to October 2023. Needs only 1-minute data.
