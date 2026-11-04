# Live dry run for Wednesday 4 November 2026: NOT FROZEN

**REAL DATA, live demonstration (n = 2 with 3 November), not evidence.** Written by an AI agent (Claude) on 4 November 2026 at about 02:16 UTC (07:46 IST), before the Nifty open at 09:15 IST (03:45 UTC).

No forecast was frozen for 4 November 2026. This is not a time problem (there were about 90 minutes left before the open). The script refused, and it was not forced.

## Why

1. **4 November 2026 is not an expiry day.** The F&O bhavcopy for 3 November 2026 (`data/raw/bhavcopy/fo_20261103.csv.gz`, REAL DATA) lists the NIFTY expiries after 3 November as 9, 17 and 23 November and 1 December 2026 (Monday 9 November is the next weekly expiry) and the next BANKNIFTY expiry as 23 November 2026. Nothing in that file expires on 4 November. (The downloader keeps only NIFTY and BANKNIFTY, so other NSE index contracts cannot be checked from it; Sensex is a BSE contract and is not in it either.)
2. **`experiments/live_dry_run.py freeze` is built for expiry days only.** `day_panel` sets the expiry dummy of the target day to True without looking it up, and `straddle_view` stops with an error when no NIFTY contract expires on the target day. Run in a scratch copy (so nothing was written here), `freeze --day 2026-11-04` stopped with: `No NIFTY contract expiring 2026-11-04 in the 2026-11-02 bhavcopy; is it an expiry day?` (exit code 1, 02:13:44 UTC).
3. **A second gap, which would block any freeze after 3 November.** `freeze` takes "the previous trading day" and the dealer gamma state from `results/pipeline_nse/daily_market.csv` and `daily_gex_panel.csv`, which end on 2 November 2026. The error message above shows it fell back to the 2 November bhavcopy, not the 3 November one. Bringing those files up to 3 November means rerunning the headline pipeline, which this check was not allowed to do.

Changing the script (look up the expiry flag, allow a missing straddle, build the previous-day state from `data/raw` instead of the pipeline folder) after seeing the refusal would have been a new, unregistered method, so it was not done.

See `reports/LIVE_DEMO_2026-11-04.md` for the rest of this check.
