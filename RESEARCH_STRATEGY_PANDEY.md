# Research strategy: Nifty expiry-day volatility and dealer gamma

Prepared for discussion with Prof. Piyush Pandey (IIT Bombay, SJMSOM). Author: Shreyan Thanki.

*Sections 1 to 8 are the plan as written before any real data were used (2 November 2026), kept unchanged so the plan can be compared with what happened. Sections 9 to 11 were rewritten on 3 November 2026 after the real-data run: what each question and hypothesis turned out to be, a scored ranking of the directions to take next, and the next actions.*

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

## 9. What happened: each question and hypothesis on real data

All REAL DATA, 2 January 2023 to 2 November 2026 (942 trading days, 200 Nifty expiries). The documents named hold the numbers and the files behind them; [docs/TEST_LEDGER.md](docs/TEST_LEDGER.md) lists all 22,586 tests with p-values corrected across the whole project (Benjamini-Yekutieli).

| Question or hypothesis | Answer | Where |
| --- | --- | --- |
| RQ1: does positioning forecast expiry-day volatility out of sample? | **No.** GEX features add nothing under any sign rule. Separately, a hybrid that corrects mcsGARCH beats it by 5.8%, 7.6% and 6.4% at 15 minutes, 30 minutes and the last hour, also before the closing auction and on a later hold-out; at 15 and 30 minutes it also beats an mcsGARCH given the same information (4.8%, 5.9%), in the last hour it does not (the gain is mostly VIX information). It replicates on Bank Nifty with nothing changed (P91: 3.9%, 5.0%). | [REAL_DATA_RESULTS.md](docs/REAL_DATA_RESULTS.md) R1, [MODEL_ITERATIONS_REAL.md](docs/MODEL_ITERATIONS_REAL.md) |
| RQ2: can the dealer sign be measured from public data? | **Not identified.** Estimating the sign from the market's response, the end-of-day outcome points to the reverse of Pro and the volatility outcome points nowhere clearly (an earlier "nearly opposite answers" is not established, docs/REFEREE_RESPONSE_F2.md section 6), and no estimated sign beats the fixed rules out of sample. The one robust GEX pattern (below) is carried by the daily Pro participant weights, not by the option chain. | [DEALER_SIGN.md](docs/DEALER_SIGN.md), [FALSIFICATION.md](docs/FALSIFICATION.md) |
| RQ3: do the paper's findings hold with dependence-aware inference and in new regimes? | **Partly.** The "after" sample replicates almost exactly; the "before" window appears to be 14 Nov 2022 to 10 Feb 2023. Nifty findings survive day-level tests; the Bank Nifty Wednesday volatility result and hour-level connectedness do not. The expiry premium depends on the regime and sits at the end of the day. | [REPLICATION_NSE.md](docs/REPLICATION_NSE.md), [EVENT_STUDY.md](docs/EVENT_STUDY.md), R2 |
| RQ4: is connectedness linked to gamma? | **Weakly at most.** The pre-stated test is borderline (p 0.044) and disappears with a VIX control; total gamma (no sign) goes with lower connectedness. | [CONNECTEDNESS_GAMMA.md](docs/CONNECTEDNESS_GAMMA.md) |
| H1: GEX adds forecasting information | Not supported, and the power study shows the test could only have seen large gains. | R1, [POWER_REAL.md](docs/POWER_REAL.md) |
| H2: the effect has the hedging sign | Not supported. The "higher GEX, calmer next day" link tracks gross gamma (the "gamma size" branch of the plan's own decision tree), and the end-of-day reversal is the opposite of the hedging prediction under the Pro sign. | [FALSIFICATION.md](docs/FALSIFICATION.md) |
| H3: participant data identify the sign | Not supported (see RQ2). | [DEALER_SIGN.md](docs/DEALER_SIGN.md) |
| H4: the paper's increases survive dependence-aware tests | Supported for Nifty, not for the Bank Nifty Wednesday result. | [REPLICATION_NSE.md](docs/REPLICATION_NSE.md) |

**Path taken through the decision tree of Section 7:** the hybrid is better than mcsGARCH; adding GEX helps under no rule; the in-sample GEX effects that exist behave like gross gamma; so the result is "a credible null on positioning, consistent with Dim, Eraker and Vilkov (2024)", plus one unexpected, robust and untradable in-sample regularity (the end-of-day reversal after negative Pro GEX).

**What did not go to plan.** NSE's closing auction from 3 August 2026 changed the last minutes of the day mid-sample and inflated one early headline; it was caught and corrected. Most in-sample GEX links failed the falsification tests. BSE's servers are not reachable from the cloud setup, so BSE Sensex expiries can only be controlled for with a reported calendar.

## 10. Candidate directions, scored 1 to 10

Each direction is scored on five criteria from 1 (poor) to 10 (excellent). The scores are our judgement from the evidence above, not a calculation, and the overall score is the plain average rounded to the nearest whole number.

* **Evidence:** how strong and robust the real-data results behind it are.
* **Novelty:** how new it is relative to the literature in [docs/LITERATURE_REVIEW.md](docs/LITERATURE_REVIEW.md) and to the paper.
* **Feasibility:** can it be finished with data we can get, in weeks rather than months.
* **Fit:** how directly it builds on Prof. Pandey's paper and interests.
* **Risk (inverted):** 10 means little risk that it collapses on new data or a referee's first question.

| Direction | Evidence | Novelty | Feasibility | Fit | Risk (inv.) | Overall | Why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D1. Expiry rules as natural experiments (event study, regimes, Sensex control, pinning by regime) | 7 | 7 | 8 | 10 | 7 | **8** | Extends the paper's own design to five rule changes with comparison indices and placebo dates; end-of-day premium follows the expiry; needs verified Sensex dates |
| D2. Pro positioning and the end-of-day reversal | 8 | 8 | 6 | 7 | 5 | **7** | Survived every falsification test and is not tradable after costs, which makes it a clean market-structure fact; what "Pro" measures is open, and the forward test will take a year or more |
| D3. Forecasting expiry-day volatility (hybrid beats mcsGARCH) | 8 | 5 | 9 | 8 | 7 | **7** | Robust and survives the project-wide correction; methodologically incremental |
| D4. Replication note for the authors (the "before" window, re-keyed printed values) | 9 | 3 | 10 | 10 | 8 | **8** | Short, checkable and directly useful to Prof. Pandey; not a paper on its own |
| D5. Identifying the dealer sign with better data | 3 | 9 | 3 | 7 | 4 | **5** | The most important open question, but public NSE data cannot answer it; needs broker-level or strike-level positioning data |
| D6. Intraday option-chain collection and intraday GEX | 2 | 6 | 5 | 6 | 5 | **5** | Minute-level gamma at spot showed nothing with end-of-day positions; real intraday positions would be new but must be collected going forward |
| D7. Connectedness by gamma regime (Nifty and Sensex next) | 3 | 7 | 6 | 9 | 3 | **6** | A natural extension of the paper's connectedness tables; the Bank Nifty version is fragile |
| D8. Pinning mechanism | 3 | 6 | 7 | 6 | 4 | **5** | Pinning is real in one regime, but none of six candidate mechanisms explains it |
| D9. Trading layer on options | 1 | 4 | 2 | 3 | 2 | **2** | The one robust pattern does not survive futures costs; an options backtest needs intraday option prices |
| D10. Sequence model for intraday volatility | 2 | 3 | 6 | 3 | 3 | **3** | Trees already capture the structure; about 100 weekly expiries since November 2024 is too few |

**Recommendation.** Lead with D1 and D4 together (they are the most natural continuation of Prof. Pandey's paper and the most certain), keep D3 as the forecasting chapter, and carry D2 as the paper's most interesting open finding, with a forward test on data after 2 November 2026 ([docs/prereg/FORWARD_TEST.md](docs/prereg/FORWARD_TEST.md), committed before any of those data existed) as its out-of-sample commitment. D5 depends on what data Prof. Pandey's group can access and is the first thing to ask him about.

## 11. Next actions

1. Send Prof. Pandey [PITCH_PANDEY.md](PITCH_PANDEY.md), with the replication question about the "before" window first.
2. Ask him about the "Pro" category and any positioning or intraday option data his group can use (D5, D6).
3. Download BSE circulars on a home machine to verify the Sensex expiry calendar (D1).
4. Collect data for the forward test (D2): it is pre-registered ([docs/prereg/FORWARD_TEST.md](docs/prereg/FORWARD_TEST.md)) and runs once per hypothesis at its stopping point (the Tuesday last-hour premium after about 100 weeks, the end-of-day reversal after 130 negative-Pro-GEX days), never before.
5. Turn the working-paper draft into the paper for whichever of D1 or D2 he prefers.
