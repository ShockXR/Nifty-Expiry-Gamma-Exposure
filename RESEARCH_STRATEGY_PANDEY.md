# Research strategy: Nifty expiry-day volatility and dealer gamma

Prepared for discussion with Prof. Piyush Pandey (IIT Bombay, SJMSOM). Author: Shreyan Thanki.

## 1. One-paragraph summary

Chhimwal, Pandey and Pandey (2025) show that after NSE separated the Nifty and Bank Nifty expiry days, each index became *more* volatile and more heavily traded on its own expiry day, while the two indices became less connected. The paper documents the effect but leaves its cause as a conjecture. A well-studied candidate cause is option dealers' hedging: when dealers are net short gamma, hedging pushes prices further in the direction they are already moving; when they are net long gamma, it pushes back. This project builds the tools to test that mechanism on Nifty, and frames the test as a forecasting contest: does knowing dealer positioning help forecast expiry-day volatility *out of sample*, beyond the paper's own mcsGARCH model? That question has an answer either way, and it uses the paper's own model as the benchmark.

## 2. What the paper contributes, and what it leaves open

See [docs/PAPER_DEEP_DIVE.md](docs/PAPER_DEEP_DIVE.md) for the full reading. In short:

| The paper does | The paper does not |
| --- | --- |
| Before/after study of an expiry-day change, on 1-minute data | Test a mechanism (no option-positioning data) |
| Fits mcsGARCH separately before and after | Forecast anything out of sample |
| Diebold-Yilmaz connectedness by sub-sample and trading hour | Report uncertainty for connectedness |
| Wilcoxon tests on per-minute volume and volatility | Use tests that allow for dependence within a day |
| Covers November 2023 to May 2024 | Cover the single weekly expiry regime (from 20 Nov 2024) or the Tuesday expiry (from 1 Sep 2025) |

## 3. Research questions, in priority order

**RQ1 (headline). Does option positioning forecast expiry-day volatility?** Out of sample, does a model with gamma exposure (GEX) features beat the same model without them, and both beat mcsGARCH, on Nifty expiry days, especially in the last hour?

**RQ2. Can the sign of dealer gamma be measured from public data?** NSE publishes only aggregate participant-wise open interest (Client, DII, FII, Pro), not who holds which strike. Which signing rule (US convention, Pro, mirror of Client) carries real information? If none do, that is a result worth reporting.

**RQ3. Do the paper's findings hold up with dependence-aware inference and on new regimes?** Re-run the before/after design with day-level tests, HAC and block bootstrap, and apply it to 20 Nov 2024 and 1 Sep 2025.

**RQ4. Is the last-hour connectedness peak linked to gamma?** Is Nifty and Bank Nifty connectedness higher on negative-GEX days?

## 4. Hypotheses and what would count as evidence

| Hypothesis | Evidence for | Evidence against |
| --- | --- | --- |
| H1: GEX adds forecasting information on expiry days | One-sided Diebold-Mariano rejects equal loss for (model without GEX) versus (with GEX) on expiry-day test days, under at least one sign rule, with the same result in the last hour | No significant loss reduction under any rule, at a sample size the power study says is adequate |
| H2: the effect has the hedging sign | Higher forecast and realized variance when signed GEX is negative, lower when positive, in partial-dependence and SHAP plots | Effect present only for unsigned (gross) gamma, which would point to a "gamma size" or attention effect rather than hedging direction |
| H3: participant data identify the sign | The participant-signed model beats the US-convention model | No difference, or the US rule wins |
| H4: the paper's volume/volatility increases survive dependence-aware tests | Day-level and bootstrap tests still reject | Only the minute-level test rejects |

## 5. What is worth validating, and what is not

**Worth doing now:**
1. Engines that are provably correct (greeks, GEX, mcsGARCH, connectedness), each checked against closed forms, finite differences or known simulated parameters. Done, 26 tests.
2. A research design that can produce a clean null. A GEX test that always "finds" something is worthless. The power study (E4) checks the false-positive rate when there is no effect and the detection rate when there is one.
3. Checks of the paper's inference at its own sample sizes (E2, E3, E5, E6). Cheap, informative, and the most direct way to show the professor we read the paper carefully.

**Not worth doing before real data:**
1. Tuning deep sequence models. With about 100 weekly expiries since November 2024, there are not enough independent expiry days for a sequence model to beat trees reliably, and the Indian daily-frequency evidence (Mahajan, Thakan and Malik 2022) does not favour them either. Kept as an alternative.
2. Trading strategy profitability claims. A backtest on synthetic data says nothing about real profits. The trading layer is kept as a design in [alternatives/ALTERNATIVES.md](alternatives/ALTERNATIVES.md) until real option prices are available.
3. Reproducing the paper's exact tables. We would need the same vendor data and the unreported VAR settings. We instead built the tools and a sensitivity grid so the replication can be run in a day once data arrive.

## 6. Method in one page

1. **Data layer.** NSE F&O bhavcopy (legacy and UDiFF layouts), participant-wise OI, 1-minute index bars, India VIX. Loaders exist and are tested on fixture files in the published layouts. Not yet tested on live files, because this build environment cannot reach NSE (see [docs/DATA_ACQUISITION.md](docs/DATA_ACQUISITION.md)).
2. **Daily GEX panel.** Black-76 implied vols from settlement prices, spot gamma, GEX per strike, net GEX under each sign rule, gamma flip, OI walls, max pain.
3. **Volatility baseline.** Our own mcsGARCH (the paper's model), with an optional expiry-day diurnal curve.
4. **Forecast ladder.** Naive, VIX, HAR, mcsGARCH, LightGBM, and a hybrid where LightGBM corrects the mcsGARCH forecast. Walk-forward, monthly refits, nothing from the future.
5. **Tests.** QLIKE and MSE losses, Diebold-Mariano on daily losses, Model Confidence Set, GEX ablation under each sign rule.

## 7. Decision tree after the first real-data run

```
Is the walk-forward hybrid better than mcsGARCH on expiry days?
├── No  -> report the paper's model as hard to beat; focus on RQ3 (inference, new regimes)
└── Yes -> Does adding GEX help (DM test, expiry days, last hour)?
           ├── Under participant sign only -> strongest result: positioning matters and is measurable
           ├── Under US sign only          -> check whether the US rule is proxying for gross gamma
           ├── Gross gamma only            -> "gamma size" effect; the sign is not identified
           └── None                        -> check power (E4 tells us the detectable effect size);
                                              a credible null, consistent with Dim, Eraker and Vilkov (2024)
```

## 8. Risks

| Risk | Plan |
| --- | --- |
| Intraday option-chain history is expensive or unavailable | All results use end-of-day GEX lagged one day, which needs only free bhavcopy. Intraday GEX is an upgrade, not a dependency. |
| Participant data are too coarse to sign gamma | Reported as RQ2's answer. The power study shows how much a noisy sign costs. |
| Few expiry days (about 100 since Nov 2024) | The power study sets expectations; the last-hour target uses one observation per day, so it is the weakest. Pool Tuesday and Thursday regimes with a regime dummy. |
| Overfitting through repeated tuning | Walk-forward only; tuning inside training windows; Model Confidence Set over the whole ladder. |
| Overlap with Prof. Pandey's other students | This project is about options positioning and volatility, not limit order books. |

## 9. What has been established so far (synthetic data only)

See [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md). Everything there is from simulated markets or from arithmetic on published table values. No real-market result is claimed anywhere in this repository.

## 10. Next actions

1. Download real data on a machine that can reach NSE (script and steps in the data guide).
2. Run `python scripts/run_pipeline.py --source nse` and check the loaders on the first real files (especially whether OI is in contracts or units).
3. Run the replication grid on the paper's window and compare with Tables 2 to 6.
4. Run the forecasting contest on November 2024 to the present and report RQ1 and RQ2.
5. Meet Prof. Pandey with [PITCH_PANDEY.md](PITCH_PANDEY.md) and the first real scoreboard.
