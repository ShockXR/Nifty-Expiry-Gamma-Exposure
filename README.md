# Nifty Expiry Gamma Exposure

**Who is short gamma on Nifty expiry day?** This project builds a dealer gamma exposure (GEX) measure for Nifty 50 index options and tests whether it explains intraday momentum and volatility on expiry day, under SEBI's single weekly expiry regime.

> Status: early prototype. The research design is set, and the code is being built.

## Motivation

Since 20 November 2024, SEBI allows each exchange only one weekly expiry on a benchmark index (circular SEBI/HO/MRD/TPD-1/P/CIR/2024/132). NSE kept Nifty 50 and dropped the other weeklies. Expiry-day activity is therefore concentrated in a single contract.

Earlier work on NSE expiry days finds that volume and volatility rise on an index's own expiry day, and that spillovers between indices peak in the last trading hour (Chhimwal, Pandey and Pandey, 2025). That work documents the effect, but it does not test the mechanism behind it.

The US literature points to one candidate: dealers' delta hedging of option positions. When dealers are net short gamma, their hedging trades go with the market and amplify moves. When they are net long gamma, their hedging trades go against the market and dampen moves (Baltussen, Da, Lammers and Martens, 2021; Dim, Eraker and Vilkov; Bandi, Fusari and Renò). We are not aware of a peer-reviewed study of this mechanism on Nifty.

## Research questions

1. **Sign:** Can the sign of dealer gamma on Nifty be identified from public NSE data, such as participant-wise open interest (Client, DII, FII, Pro)?
2. **Momentum:** Does GEX predict last-hour returns on expiry day, conditional on the return earlier in the day?
3. **Volatility:** Does GEX explain intraday volatility beyond the usual time-of-day pattern?
4. **Regime:** Did these relationships change after the single weekly expiry rule took effect?

## Method

| Step | What | Notes |
| --- | --- | --- |
| 1 | Build GEX from the option chain | Sum of gamma times open interest times contract multiplier, signed by assumed dealer side. Report by strike and in aggregate. |
| 2 | Participant-signed GEX | Use NSE participant-wise OI to sign positions instead of assuming dealers are short customer demand. |
| 3 | Volatility baseline | Fit a multiplicative component GARCH (mcsGARCH; Engle and Sokalska, 2012) to 1-minute returns. Use its de-seasonalised residual as the target. |
| 4 | Momentum test | Regress last-hour return on earlier intraday return, interacted with GEX sign and size. |
| 5 | Event study | Compare the periods before and after 20 November 2024. |
| 6 | Robustness | HAC or block-bootstrap inference, placebo non-expiry days, alternative signing rules. |

## Data

| Source | Use | Access |
| --- | --- | --- |
| NSE F&O bhavcopy | End-of-day option OI, prices, strikes | Free, daily download |
| NSE participant-wise OI | Client, DII, FII, Pro positions | Free, daily CSV |
| Nifty 50 1-minute bars | Intraday returns and volatility | Free or low cost vendors |
| Intraday Nifty option chain | Intraday GEX | Main data risk. Fallback: end-of-day GEX applied to the next session. |

Raw data is not committed. See [`data/README.md`](data/README.md).

## Repository layout

```
.
├── data/          raw and processed data (git-ignored)
├── docs/          notes and write-ups
├── notebooks/     exploratory analysis
├── src/niftygex/  package code
│   ├── data.py         loaders for NSE files and intraday bars
│   ├── greeks.py       Black-76 gamma and implied volatility
│   ├── gex.py          gamma exposure by strike and in aggregate
│   └── volatility.py   mcsGARCH baseline
└── tests/
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Roadmap

- [ ] Data loaders for bhavcopy and participant-wise OI
- [ ] Black-76 gamma and end-of-day GEX by strike
- [ ] GEX time series across the 20 Nov 2024 regime change
- [ ] mcsGARCH baseline on 1-minute Nifty returns
- [ ] Last-hour momentum regression conditional on GEX
- [ ] Short write-up of first results

## References

- Baltussen, G., Da, Z., Lammers, S. and Martens, M. (2021). Hedging demand and market intraday momentum. *Journal of Financial Economics*, 142(1), 377 to 403.
- Bandi, F. M., Fusari, N. and Renò, R. 0DTE option pricing. Working paper.
- Chhimwal, B., Pandey, V. and Pandey, P. (2025). Effect of multiple index derivative expiry on volatility, volume, and connectedness: a tale of two stock indices in India. *Review of Derivatives Research*, 28, 14. doi:10.1007/s11147-025-09221-8
- Dim, C., Eraker, B. and Vilkov, G. 0DTEs: Trading, gamma risk and volatility propagation. Working paper.
- Engle, R. F. and Sokalska, M. E. (2012). Forecasting intraday volatility in the US equity market: multiplicative component GARCH. *Journal of Financial Econometrics*, 10(1), 54 to 83.
- SEBI (2024). Measures to strengthen equity index derivatives framework for increased investor protection and market stability. Circular SEBI/HO/MRD/TPD-1/P/CIR/2024/132, 1 October 2024.

## Author

Shreyan Thanki ([@ShockXR](https://github.com/ShockXR))
