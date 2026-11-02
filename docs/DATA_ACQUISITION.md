# Getting the real data

**Status (2 November 2026):** everything below now runs on a cloud machine. NSE's file archive server (`nsearchives.nseindia.com`) is reachable from the project's cloud environment, even though the main website (`www.nseindia.com`) is not, and none of the files we need require the main website. Free 1-minute index bars come from two public GitHub datasets, so no broker account is needed.

Four commands rebuild the full real-data set from scratch (about 25 minutes, almost all of it polite downloading):

```bash
python scripts/download_nse.py --start 2023-01-01 --end 2026-11-02 --out data/raw
python scripts/fetch_minute_bars.py --out data/raw
python scripts/validate_minute_bars.py --raw data/raw --start 2023-01-01
python scripts/run_pipeline.py --source nse --raw data/raw
```

Raw data stay out of git (`data/raw/` is ignored). Derived outputs (GEX panel, forecasts, scores, validation tables) are committed under `results/`.

## What is downloaded

| Dataset | Saved as (under `data/raw/`) | Source | Used for |
| --- | --- | --- | --- |
| F&O bhavcopy, one file per day | `bhavcopy/fo_YYYYMMDD.csv.gz` | NSE archive, free | option chains, OI, daily GEX, the real expiry calendar |
| Participant-wise OI, one file per day | `participant_oi/fao_participant_oi_DDMMYYYY.csv` | NSE archive, free | sign rules |
| Daily index values, one file per day | `index_close/ind_close_all_DDMMYYYY.csv`, combined into `index_daily.csv` | NSE archive, free | India VIX, official Nifty 50 and Nifty Bank open, high, low, close, and the official list of trading days |
| India VIX daily close | `india_vix.csv` (built from the above) | NSE archive, free | VIX model and feature |
| Nifty 50 1-minute bars | `nifty_1min.csv` | GitHub: technovusin/nifty50-historical-data (MIT licence) | returns, targets |
| Nifty Bank 1-minute bars | `banknifty_1min.csv` | GitHub: sandeepkapri/BankNifty-Data (MIT licence) | paper replication |

The bhavcopy is filtered on download to NIFTY and BANKNIFTY index options and futures (the full file is about 6 MB a day; the filtered one about 0.1 MB). The columns are kept exactly as NSE publishes them.

### NSE file locations (all on `https://nsearchives.nseindia.com`)

* UDiFF F&O bhavcopy (from 8 July 2024): `/content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip`
* Legacy F&O bhavcopy (before 8 July 2024): `/content/historical/DERIVATIVES/YYYY/MON/foDDMONYYYYbhav.csv.zip`
* Participant-wise OI: `/content/nsccl/fao_participant_oi_DDMMYYYY.csv`
* All indices, daily: `/content/indices/ind_close_all_DDMMYYYY.csv`

## Facts checked on the real files

1. **Open interest is in units (shares), not contracts, in both layouts.** On 30 October 2025, adding up OI divided by each row's lot size over all index options gives 4,230,409 call contracts and 3,261,187 put contracts, exactly the TOTAL "Option Index Call Long" and "Option Index Put Long" in that day's participant-wise file. Index futures match exactly on 10 January 2024 as well (433,529 contracts).
2. **Two lot sizes can be live on the same day** (NIFTY 75 and 65 in late 2025, for different expiries), so a single lot size per day is wrong around lot changes. The pipeline therefore keeps OI in shares and uses a lot size of 1 (`--oi-unit keep_units`, the default for `--source nse`), which makes GEX exact in rupees per 1% move.
3. **On expiry days the two files disagree, and we know why.** On a day when some index options expire, the bhavcopy still shows the expiring contracts' open interest, while the participant file has already dropped them. Leaving out contracts that expire that day makes the two agree to within one contract on all 9 dates checked (January 2024 to October 2026, with and without expiries; `scripts/check_oi_units.py`, output in `results/data_validation/oi_units_check.csv`). The pipeline never uses a contract on its own expiry day (each day's chain is the next expiry strictly after that day), so this does not affect GEX.
4. **Expiry days** are read from the bhavcopies (a date is an expiry day if contracts expiring that day trade on it). This catches holiday shifts, for example the Wednesday 25 January 2023 expiry before the Republic Day holiday.
5. **The minute bars match NSE.** See `results/data_validation/summary.csv` and the real-data results document for the numbers. Highs and lows are compared exactly. Closes are not expected to match: NSE's official close is an average over the last 30 minutes, not the last trade.

## 1-minute bars: limits

* No minute volume for the index before 2026, so the volume and order-flow features are switched off (constant) on real data before then.
* Nifty Bank bars end on 22 April 2026. They are used only for the paper replication (November 2023 to May 2024).
* These are third-party files. They are checked against NSE every day before use, and any day that fails is listed.

## Session grid

The loader keeps 09:15 to 15:29 bar start times (375 bars). Days with less than 95% of the bars (Muhurat and other special short sessions) and days that are not in NSE's own list of trading days are dropped.
