# Nifty Expiry Gamma Exposure

**Does option positioning help forecast Nifty expiry-day volatility?** This project rebuilds the volatility model from Chhimwal, Pandey and Pandey (2025) as an out-of-sample forecasting baseline, adds dealer gamma exposure (GEX) built from NSE option data, and tests whether positioning carries information the baseline misses, especially in the last hour of expiry day.

> **Status (3 November 2026): running on REAL NSE data.** 942 trading days (January 2023 to 2 November 2026) of NSE F&O bhavcopies, participant-wise open interest and official index values including India VIX, from NSE's archive server, plus free third-party 1-minute Nifty, Bank Nifty and Fin Nifty bars that we checked day by day against NSE's official values. Simulation results (used to validate every engine first) are kept separate and labelled. Start with [docs/REAL_DATA_RESULTS.md](docs/REAL_DATA_RESULTS.md).

**Real-data results so far** (details, caveats and the file behind every number are in the linked documents):

* **Forecasting (out of sample, 792 test days;** [docs/MODEL_ITERATIONS_REAL.md](docs/MODEL_ITERATIONS_REAL.md)**).** A hybrid that corrects the paper's mcsGARCH with a LightGBM model of its errors forecasts the next 15 minutes, the next 30 minutes and the last hour 5.8%, 7.6% and 6.4% better (QLIKE loss, Diebold-Mariano p < 0.001), against an mcsGARCH refitted as often as every other model. It also holds before NSE's closing auction began (3 August 2026) and on a later hold-out. At 15 and 30 minutes it survives a correction for trying 31 models and beats an mcsGARCH given the same information (4.8% and 5.9%, an edge that comes from the periods since November 2024); in the last hour it does not (an mcsGARCH given India VIX comes within 1.7%; [docs/REFEREE_RESPONSE_FORECAST.md](docs/REFEREE_RESPONSE_FORECAST.md)). A registered ablation ([docs/P106_GAIN_DECOMPOSITION.md](docs/P106_GAIN_DECOMPOSITION.md)) finds the gain comes from recent realized volatility (34% and 38%) and the time of day (23% and 25%), not from the option market: removing the option-implied, VIX or calendar inputs costs nothing, and the edge over the information-matched mcsGARCH appears after November 2024 mainly because that benchmark got worse. Run unchanged on Bank Nifty under a registered plan, the hybrid is 3.9% and 5.0% better at 15 and 30 minutes ([docs/BANKNIFTY_FORECAST.md](docs/BANKNIFTY_FORECAST.md)), and on Fin Nifty, under a second registered plan, 4.9% and 6.1% better ([docs/THIRD_INDEX_FORECAST.md](docs/THIRD_INDEX_FORECAST.md)); pooled with errors that allow for the indices moving together, the three count as about two independent replications ([docs/P107_POOLED_THREE_INDEX.md](docs/P107_POOLED_THREE_INDEX.md)). An earlier apparent last-hour edge for HAR came from the closing-auction days and was withdrawn. Adding dealer gamma (GEX) features does not improve Nifty forecasts under any of three sign rules (on Bank Nifty they helped slightly, only in the 09:15 forecast; on Fin Nifty they did not help).
* **Dealer gamma, after falsification** ([docs/FALSIFICATION.md](docs/FALSIFICATION.md), [docs/DEALER_SIGN.md](docs/DEALER_SIGN.md)). One in-sample pattern survived most attempts to break it (not a correction over all 742 tests of that study): Nifty tends to reverse in the last 30 minutes, and only after evenings when GEX built from NSE's "Pro" participant positions was negative (the opposite of the US hedging-momentum story). The "higher GEX, calmer next day" link turned out to track the total amount of option gamma, not its sign, and the expiry-day sign flip is in-sample only. The proxy-free version of the hedging story (dealers short all options, so the close continues the day's move) fails: the close reverses instead ([docs/REFEREE_RESPONSE_F2.md](docs/REFEREE_RESPONSE_F2.md)). Estimating the dealer sign from the data, the end-of-day outcome points to the reverse of the Pro positions and the volatility outcome points nowhere clearly, so who the dealers are remains the open question.
* **Expiry rules as natural experiments** ([docs/EVENT_STUDY.md](docs/EVENT_STUDY.md)). The whole-day expiry premium barely moves with NSE's rule changes; the end-of-day premium follows the expiry, although part of it is one-minute bid-ask noise and the placebo p-values rest on few independent windows ([docs/REFEREE_RESPONSE_EXPIRY.md](docs/REFEREE_RESPONSE_EXPIRY.md)).
* **Replication of the 2025 paper** ([docs/REPLICATION_NSE.md](docs/REPLICATION_NSE.md)). The "after" sample reproduces the printed tables closely. The stated "before" window does not, while a 64-day window one year earlier reproduces the printed "before" numbers. Two independent searches put its start between 14 and 23 November 2022 and its end between 10 and 21 February 2023 ([docs/P108_BLIND_WINDOW.md](docs/P108_BLIND_WINDOW.md)); we will ask the authors which dates they used.
* **Pinning** ([docs/PINNING_STUDY.md](docs/PINNING_STUDY.md)). The expiry-day close sits closer to a strike only in some periods (Nifty from November 2024 to August 2025, Bank Nifty while it had weekly options), and our dealer-gamma proxies do not explain when.

## How this work was done

Shreyan directed the project; the code, analyses and documents were produced with extensive AI assistance (Claude, by Anthropic), and most commits list Claude as author. The checks described as independent (the referee-style review in `docs/INDEPENDENT_REVIEW.md`, the blind re-key in `docs/REKEY_CHECK.md` and later referee reports) were done by separate AI agents, not by people. Every number comes from code and data in this repository and can be regenerated with `python scripts/reproduce_all.py` (see `docs/REPRODUCIBILITY.md`). "Pre-registered" here means an analysis plan committed to git before that study's results were computed; only the forward test (`docs/prereg/FORWARD_TEST.md`) was committed before its data existed.

## Why this question

Chhimwal, Pandey and Pandey (2025) show that after NSE gave Nifty and Bank Nifty different expiry days, each index became more volatile and more heavily traded on its own expiry day, and the two became less connected. They suggest unwinding and retail flows as causes but do not test them.

A well-studied mechanism is option dealers' hedging. When dealers are net short gamma, hedging pushes the index further in the direction it is moving; when net long gamma, it pushes back (Ni, Pearson, Poteshman and White 2021; Baltussen, Da, Lammers and Martens 2021). Gamma is largest for near-the-money options close to expiry, so if the mechanism matters anywhere on Nifty, it should matter on expiry day. We found no peer-reviewed test of it on NSE options.

## What is in the box

| Part | Module | Validated by |
| --- | --- | --- |
| Black-76 prices, greeks (delta, gamma, vega, vanna, charm) and IV solver | `niftygex.greeks` | parity, finite differences, IV round trip on 5,000 options including expiry-day ones |
| GEX per strike and net, under five sign rules; gamma flip; OI walls; max pain | `niftygex.gex` | hand formula, finite-difference definition of GEX, bounds |
| Rogers-Satchell, realized variance, bipower variation, bulk volume classification | `niftygex.realized` | Brownian simulation |
| mcsGARCH (the paper's model) with an optional expiry-day diurnal curve | `niftygex.volatility` | parameter recovery on simulated data, causality of forecasts |
| Diebold-Yilmaz connectedness, block-bootstrap intervals | `niftygex.connectedness` | closed form, independent series, VAR recovery |
| Day-level, HAC and bootstrap before/after tests | `niftygex.inference` | experiment E2 |
| Expiry calendar (Thursday, then Tuesday from 1 Sep 2025) and regimes | `niftygex.expiry_calendar` | known dates, holiday roll-back |
| NSE loaders (bhavcopy legacy and UDiFF, participant OI, minute bars, VIX) | `niftygex.data` | round trip through files in NSE layouts |
| Synthetic market with a switchable dealer-gamma effect | `niftygex.simulate` | used for all validation |
| Forecast ladder, walk-forward, QLIKE/MSE, Diebold-Mariano, Model Confidence Set, GEX ablation | `niftygex.forecast`, `niftygex.evaluation` | experiment E4 |

542 tests, run with `pytest` (also on every pull request through GitHub Actions); 7 of them check real Bank Nifty and Fin Nifty data and are skipped when `data/raw` is absent.

## Documents

| Document | What it is |
| --- | --- |
| [RESEARCH_STRATEGY_PANDEY.md](RESEARCH_STRATEGY_PANDEY.md) | Research questions, hypotheses, decision tree, risks |
| [PITCH_PANDEY.md](PITCH_PANDEY.md) | One-page pitch for Prof. Pandey |
| [docs/PAPER_DEEP_DIVE.md](docs/PAPER_DEEP_DIVE.md) | Close reading of the 2025 paper, with checks of its published numbers |
| [docs/LITERATURE_REVIEW.md](docs/LITERATURE_REVIEW.md) | 41 checked references (36 fully verified) and what each means for the design |
| [docs/math/](docs/math/) | Derivations: greeks and GEX, mcsGARCH, connectedness, inference, forecast evaluation |
| [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md) | Every experiment, its question, its result and its limits |
| [ITERATION_LOG.md](ITERATION_LOG.md) | What went wrong during the build and what changed |
| [docs/REAL_DATA_RESULTS.md](docs/REAL_DATA_RESULTS.md) | **Real data:** forecasting contest, expiry effects by regime, GEX regressions, end-of-day momentum (R0 to R5) |
| [docs/REPLICATION_NSE.md](docs/REPLICATION_NSE.md) | **Real data:** the paper's Tables 1 to 6 rebuilt from NSE data, with robust versions of each test |
| [docs/PINNING_STUDY.md](docs/PINNING_STUDY.md) | **Real data:** do Nifty and Bank Nifty close at option strikes on expiry day? |
| [docs/EVENT_STUDY.md](docs/EVENT_STUDY.md) | **Real data:** expiry rule changes as natural experiments, with comparison indices and placebo dates |
| [docs/FALSIFICATION.md](docs/FALSIFICATION.md) | **Real data:** seven families of tests that try to break the GEX findings |
| [docs/DEALER_SIGN.md](docs/DEALER_SIGN.md) | **Real data:** estimating the dealer sign from the market's response (a negative result) |
| [docs/INTRADAY_GAMMA.md](docs/INTRADAY_GAMMA.md) | **Real data:** expiry-day gamma re-priced minute by minute |
| [docs/INDEPENDENT_REVIEW.md](docs/INDEPENDENT_REVIEW.md) | A referee-style review of the real-data results, and what changed because of it |
| [results/live/](results/live/) | **Real data:** a forecast frozen before the 3 November 2026 expiry, scored after the close |
| [notebooks/01_real_data_tour.ipynb](notebooks/01_real_data_tour.ipynb) | **Real data:** a guided tour with charts (option chain, GEX profile, an expiry day, the scoreboard) |
| [docs/DATA_ACQUISITION.md](docs/DATA_ACQUISITION.md) | How the real data are downloaded and checked |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Exact commands, versions and run times to regenerate every result |
| [alternatives/ALTERNATIVES.md](alternatives/ALTERNATIVES.md) | Trading layer, sequence model and other directions kept out of the main package |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest

# Full pipeline on a synthetic market (about 15 minutes on 4 cores)
python scripts/run_pipeline.py --source synthetic --days 500 --kappa 0.5

# Paper checks and simulation experiments
python experiments/paper_checks.py
python experiments/inference_checks.py
python experiments/power_study.py --reps 8 --days 500

# Real data (see docs/DATA_ACQUISITION.md)
python scripts/download_nse.py --start 2023-01-01 --end 2026-11-02 --out data/raw
python scripts/fetch_minute_bars.py
python scripts/validate_minute_bars.py
python scripts/run_pipeline.py --source nse          # about 8 minutes on 4 cores (the five-seed model study, --n-seeds 5, takes about 100 to 115 minutes on one core)
python experiments/real_day_level.py
python experiments/momentum_gex.py
python experiments/replicate_paper.py
python experiments/pinning_study.py
```

## Repository layout

```
.
├── src/niftygex/      package code (see table above)
├── scripts/           run_pipeline.py (one command, all outputs), download_nse.py, fetch_minute_bars.py, validation checks
├── experiments/       simulation experiments (E1 to E7) and real-data studies (real_day_level, momentum_gex, replicate_paper, pinning_study)
├── results/           outputs; real-data folders are pipeline_nse_v2 (pipeline_nse is the archived first run), real_day_level, momentum_gex, replication_nse, pinning, data_validation
├── notebooks/         real-data demo notebook
├── docs/              deep dive, literature review, math notes, experiment report, data guide
├── alternatives/      directions deliberately kept out of the main package
├── tests/             pytest suite
└── data/              raw and processed data (git-ignored)
```

## Data policy

Raw vendor and NSE files are not committed. The 2025 paper is paywalled and is not in this repository; it is cited by DOI. Derived outputs (GEX panels, forecasts, scores) can be shared.

## Key references

Full list with verification status in [docs/LITERATURE_REVIEW.md](docs/LITERATURE_REVIEW.md).

- Chhimwal, B., Pandey, V. and Pandey, P. (2025). Effect of multiple index derivative expiry on volatility, volume, and connectedness: a tale of two stock indices in India. *Review of Derivatives Research*, 28(3), 14. doi:10.1007/s11147-025-09221-8
- Engle, R. F. and Sokalska, M. E. (2012). Forecasting intraday volatility in the US equity market: multiplicative component GARCH. *Journal of Financial Econometrics*, 10(1), 54 to 83. doi:10.1093/jjfinec/nbr005
- Ni, S. X., Pearson, N. D., Poteshman, A. M. and White, J. (2021). Does option trading have a pervasive impact on underlying stock prices? *Review of Financial Studies*, 34(4), 1952 to 1986. doi:10.1093/rfs/hhaa082
- Baltussen, G., Da, Z., Lammers, S. and Martens, M. (2021). Hedging demand and market intraday momentum. *Journal of Financial Economics*, 142(1), 377 to 403. doi:10.1016/j.jfineco.2021.04.029
- Dim, C., Eraker, B. and Vilkov, G. (2024). 0DTEs: Trading, gamma risk and volatility propagation. Working paper, SSRN 4692190.
- Diebold, F. X. and Yilmaz, K. (2012). Better to give than to receive. *International Journal of Forecasting*, 28(1), 57 to 66. doi:10.1016/j.ijforecast.2011.02.006
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246 to 256. doi:10.1016/j.jeconom.2010.03.034

## Author

Shreyan Thanki ([@ShockXR](https://github.com/ShockXR))
