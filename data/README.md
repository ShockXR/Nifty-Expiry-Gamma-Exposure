# Data

Raw data are not committed to this repository. See `docs/DATA_ACQUISITION.md` for the commands that rebuild them (all free sources, about 25 minutes).

Layout after the download scripts run:

```
data/
└── raw/
    ├── bhavcopy/            NSE F&O bhavcopy, one gzip CSV per trading day (NIFTY and BANKNIFTY index options and futures)
    ├── participant_oi/      NSE participant-wise open interest, one CSV per day
    ├── index_close/         NSE daily index values, one CSV per day
    ├── index_daily.csv      the above combined (Nifty 50, Nifty Bank, India VIX)
    ├── india_vix.csv        India VIX daily close
    ├── nifty_1min.csv       Nifty 50 1-minute bars (third-party, validated against NSE)
    ├── banknifty_1min.csv   Nifty Bank 1-minute bars (third-party, validated against NSE)
    └── minute_bars_source.json   where the minute bars came from (repository and commit)
```
