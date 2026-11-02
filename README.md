# Nifty Expiry Gamma Exposure

**Does option positioning help forecast Nifty expiry-day volatility?** This project rebuilds the volatility model from Chhimwal, Pandey and Pandey (2025) as an out-of-sample forecasting baseline, adds dealer gamma exposure (GEX) built from NSE option data, and tests whether positioning carries information the baseline misses, especially in the last hour of expiry day.

> **Status (November 2026): working prototype, validated on synthetic data only.** Every engine is tested against closed forms or simulated data with known answers, and the full pipeline runs from NSE-format files to a forecast scoreboard. It has not yet been run on real market data, because the machine it was built on cannot reach NSE. No result in this repository is a real-market result. See [docs/DATA_ACQUISITION.md](docs/DATA_ACQUISITION.md) to run it on real data.

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

31 tests, run with `pytest`.

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
| [docs/DATA_ACQUISITION.md](docs/DATA_ACQUISITION.md) | How to get the real data, and checks for the first real files |
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

# Real data, once downloaded (see docs/DATA_ACQUISITION.md)
python scripts/download_nse.py --start 2023-01-01 --end 2026-10-31 --out data/raw
python scripts/run_pipeline.py --source nse --raw data/raw
```

## Repository layout

```
.
├── src/niftygex/      package code (see table above)
├── scripts/           run_pipeline.py (one command, all outputs), download_nse.py
├── experiments/       paper_checks.py (E5, E6), inference_checks.py (E2, E3), power_study.py (E4)
├── results/           outputs of the experiments and the synthetic pipeline run (all synthetic)
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
