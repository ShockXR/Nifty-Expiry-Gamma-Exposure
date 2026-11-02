# Getting the real data

The cloud machine that built this prototype cannot reach nseindia.com, nsearchives.nseindia.com, BSE, Yahoo Finance or broker APIs (its network policy blocks them; Python packages install fine). So every result in this repository so far is synthetic. The real-data path is built and tested on files written in NSE's layouts, but it has not yet seen a live NSE file. Run the steps below on your own machine.

## What is needed

| Dataset | Folder / file under `data/raw/` | Source | Used for |
| --- | --- | --- | --- |
| F&O bhavcopy, one file per day | `bhavcopy/*.csv` | NSE archives, free | daily GEX |
| Participant-wise OI, one file per day | `participant_oi/fao_participant_oi_DDMMYYYY.csv` | NSE archives, free | sign rules |
| Nifty 50 1-minute bars | `nifty_1min.csv` (timestamp, open, high, low, close) | broker API or vendor | returns, targets |
| Nifty futures 1-minute bars (optional) | `nifty_fut_1min.csv` (with volume) | broker API or vendor | volume and order flow |
| India VIX daily close | `india_vix.csv` (date, close) | NSE, free | VIX model and feature |

Sample to aim for: January 2023 to the present. That covers the paper's window (Nov 2023 to May 2024), the single weekly expiry rule (20 Nov 2024) and the Tuesday expiry (from 1 Sep 2025).

## NSE file locations

`scripts/download_nse.py` downloads bhavcopy and participant files for a date range, politely (one request per second, browser-like headers, skips files already present). It uses these URL patterns, which were correct as of 2025 but have changed before; if a download fails, check the NSE "All Reports" page for the current location.

* UDiFF F&O bhavcopy (from July 2024): `https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip`
* Legacy F&O bhavcopy (before July 2024): `https://nsearchives.nseindia.com/content/historical/DERIVATIVES/YYYY/MON/foDDMONYYYYbhav.csv.zip`
* Participant-wise OI: `https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_DDMMYYYY.csv`

The script was written for your machine and has not been run from the build environment.

## 1-minute bars

NSE does not publish free historical 1-minute index bars. Options: a broker API with historical data (for example Zerodha Kite Connect), or a data vendor. Check the current terms and cost before subscribing. Save as CSV with a timestamp column in IST and columns open, high, low, close (and volume for futures).

## Checks to run on the first real files

1. **OI units.** Open one UDiFF bhavcopy and compare `OpnIntrst` for an ATM Nifty option with the OI shown on NSE's option chain page for that day. If the file's number is the chain's number times the lot size, OI is in units: run with `--oi-unit units` (default). If equal, use `--oi-unit contracts`.
2. **Lot size.** Nifty's lot size has changed several times. The loader reads `NewBrdLotQty` from UDiFF files; for legacy files pass `--lot-size`.
3. **Session grid.** The loader keeps 09:15 to 15:29 bar start times (375 bars). The paper appears to use 360 minutes per day; the replication script should trim to match once the trimming rule is known.
4. **Holidays and special sessions.** The expiry calendar uses the trading days found in the minute bars, so holidays roll expiries back automatically. Muhurat (special) sessions should be dropped from the minute file.

## Then

```bash
python scripts/run_pipeline.py --source nse --raw data/raw
```

Outputs go to `results/pipeline_nse/`. Raw data stay out of git (`data/raw/` is ignored); derived outputs (GEX panel, forecasts, scores) can be committed.
