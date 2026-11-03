# First-run diagnostics (archived 3 November 2026)

**REAL DATA.** These files are the versions of `results/closing_auction/r1_*.csv` and
`results/forecast_diagnostics/*.csv` (and their stdout logs) that were committed before the full
reproduction of 3 November 2026. They were computed from the **first** forecasting run
(`results/pipeline_nse/`: mcsGARCH refitted every 63 days, no option-implied features, single-seed
LightGBM, model names such as `lgbm_gex` and `hybrid_gex`), except `largest_mcsgarch_errors.csv`,
which came from an earlier, superseded pass of the second run.

Since commit bf6edc0 the scripts `experiments/closing_auction_check.py` and
`experiments/forecast_diagnostics.py` read the second run (`results/pipeline_nse_v2/`), and the
reproduction rewrote the files in `results/closing_auction/` and `results/forecast_diagnostics/`
accordingly. These archived copies are kept because two documents cite first-run numbers from them:

* the withdrawn first-run claim that HAR was about 11% better than mcsGARCH in the last hour, and
  that most of it came from the closing-auction days (`closing_auction/r1_dm_tests.csv`,
  `closing_auction/r1_scoreboards.csv`, `forecast_diagnostics/tails_and_last_hour_tests.csv`);
* `docs/INDEPENDENT_REVIEW.md`, a review of the first run, rows 20 and 21 of its number check.

The scripts that made them are in the git history at commit b2bfcc9 (closing auction) and 7e57de4
(forecast diagnostics). Nothing here is used by any current result.
