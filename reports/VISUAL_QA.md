# Visual QA (Pandey prototype)

Written by an AI agent (Claude), 3 November 2026, validation agent 3. Labels: REAL = real NSE data, OUR CHECK = recomputed by me in this session.

## Summary for a beginner (5 lines)

1. I looked at the charts a professor will see: the live demo pages, the deck chart, the forecasting scoreboards and the main study figures (about 15 viewed, 96 PNG files inventoried).
2. The numbers on the charts are right: I recomputed the demo day's scores, the 8-of-25 window count and the 5.8% overall gain from the committed forecast file and they match.
3. The study figures were honest (losses shown, REAL DATA stated, wide intervals shown), but the forecasting scoreboards were hard to read: labels collided, no REAL or SYNTHETIC tag, no number of test days, and the colours disagreed with the demo.
4. I fixed those in `src/niftygex/plots.py` and redrew 62 figures from committed CSV files (same file names, same numbers). Five "diurnal curve" pictures could not be redrawn because the raw data is not in the repository.
5. Open problems: six of the fourteen deck slides may run past the bottom edge (I cannot see the real deck viewer), and the demo page chart is too small on a phone.

## What I checked and how

| Check | Method | Result |
|---|---|---|
| Inventory | `find` over `results/`, `docs/`; which script writes which figure | 96 PNG files; `plots.py` is called only by `scripts/run_pipeline.py` (scoreboards, ablations, GEX, diurnal); study figures come from `experiments/*.py`; README and PITCH embed none, `docs/` embed about 20, the deck uses one (the deck chart) |
| Labels in code | AST scan of every figure function for REAL or SYNTHETIC strings | pipeline figures lacked them (fixed); study figures state REAL DATA (a few scan hits were false alarms because the label is built from a variable; checked 3 by eye) |
| Demo page numbers | recomputed from `results/pipeline_nse_v2/forecasts_next15.csv` | QLIKE 0.1664 and 0.1554, first window 0.8498 and 0.3945, 8 of 25 windows, 565 of 792 days, all-days gain 5.7988% (OUR CHECK, REAL): all equal the page |
| Demo pages reproducible | reran the three pages into a scratch folder and `cmp` | byte-identical to `results/demo/*.html` |
| Deck table numbers | scoreboards of the three runs | rows 3 to 6 equal (5.8, 7.6, 6.4 for Nifty; 3.9, 5.0, 8.1 Bank Nifty; 4.9, 6.1, 1.0 Fin Nifty); row 2 equals P116's 4.43, 6.64, 6.16; row 1 (P118, 5.1 and 7.1) not rechecked by me |
| Browser | Chromium 1280x800 and 390x844, network off | demo pages: no console errors, no external requests, no horizontal scroll |
| Deck layout | each slide rendered in a 1920x1080 stage with the real fonts (downloaded) | see table; renderer of the real deck is unknown |

## Findings

| Figure or page | Issue | Severity | Status | Evidence |
|---|---|---|---|---|
| `results/pipeline_*/figures/scoreboard_*.png` (about 30 files) | No REAL or SYNTHETIC tag; no number of test days; "all days" silently mixes the development part (462 days) and the hold-out part (330 days) of the 792 test days; negative value labels ran into the model names ("-5.4%" glued to "hybridiv_gex_participant"); mcsGARCH drawn orange and others blue, while the demo page and deck draw mcsGARCH blue and the hybrid orange | Medium | FIXED: `plots.scoreboard_bars` now tags the data, prints the day counts and the development and hold-out split, puts negative labels right of the zero line, uses blue for mcsGARCH, orange for hybrids and grey for others; 62 figures redrawn from the committed CSVs by `reports/figure_regen/regen_pipeline_figures.py`, names unchanged | old view vs new view of `scoreboard_next15_all.png`; labels equal the CSV (+238.1, +6.1, -5.8, -6.0 checked) |
| `scoreboard_*` | The bars are dominated by the naive models (+238%), so the 5 to 6% differences that matter are a few pixels wide | Low | OPEN (linear scale kept: honest, but weak for a slide). Use the CSV table or the demo page for the 5.8% claim | `results/pipeline_nse_v2/figures/scoreboard_next15_all.png` |
| `ablation_*.png` (27 files) | Legend overlapped the title; "p<0.05" filled markers are one-sided unadjusted p-values and the figure did not say so; no sample size | Low | FIXED (layout, footer with day counts and "unadjusted") | `results/pipeline_nse_v2/figures/ablation_next15_hybrid.png` (all markers open: no GEX set is significant) |
| `gex_by_sign_rule.png` (5 files) | No data label other than "NSE" in the title; unexplained row names; each panel has its own scale | Low | FIXED (label, footer sentence, row names with spaces) | `results/pipeline_nse_v2/figures/gex_by_sign_rule.png` |
| `diurnal_curves.png` (5 files) | The 09:15 value is 25 to 30 times the average, so the rest of the day is a flat line; no data label; the title says in-sample but nothing explains why | Medium | PARTLY FIXED: the code now cuts the axis, prints the off-scale value, labels the data and adds an in-sample note (tested on a dummy curve only). BLOCKED for the committed PNGs: the fitted curve is not saved in any committed file and `data/raw` is absent | `src/niftygex/plots.py::diurnal_curves` |
| `scripts/run_pipeline.py` titles | "20-day" is typed in the title while `plots.py` typed 20 as well (two places, no shared name); scoreboard titles carry no data label | Low | PARTLY FIXED: `plots.GEX_SCALE_WINDOW` now holds the 20 and the redraw uses it. `run_pipeline.py` (not mine) should pass `data_label=` and use the constant | `reports/figure_regen/regen_pipeline_figures.py` |
| Deck slides `scope, replication, forecast-limits, expiry, trust, asks` | In a 1920x1080 stage their content ends at 1182, 1136, 1187, 1144, 1160 and 1237 px, i.e. 56 to 157 px below the slide edge; on `scope, fixes, forecast-limits, trust` the source line overlaps the cards | Medium | PARTIALLY VERIFIED: my stage may differ from the real deck viewer (section height is not fixed in the source). Check these six slides in the real viewer before the meeting. Deck sources are read-only to me | `reports/screenshots/deck_scope.png`, `deck_asks.png`, `deck_forecast-limits.png`, `deck_trust.png` (stage cuts at 1080), `deck_forecast.png` and `deck_oneday.png` fit |
| Deck slide `oneday` and `results/demo/deck_chart_2025-12-23.png` | none: axis units stated (variance times 10^-6), REAL DATA badge, legend, 8 of 25 and 0.850 vs 0.395 match the data, day chosen by a stated rule | none | VERIFIED | `results/demo/deck_chart_2025-12-23.png` |
| Demo pages (`results/demo/*.html`, scripts/demo_forecast_day.py) | Main chart text shrinks to about 6 px at 390 px width | Low | OPEN (laptop or projector is fine) | `reports/screenshots/demo_2025-12-23_phone_full.png` |
| Demo pages | Caption template says "about as busy than usual" when the day is near the median (seen with `--date 2026-07-15`) | Low | OPEN, owner of `scripts/demo_forecast_day.py` | stdout of `--date 2026-07-15` |
| Demo `--pick` vs page | `--pick` prints "gain +6.6%" (positive is better) while the page prints "hybrid vs mcsGARCH -6.6%" (negative is better) | Low | OPEN, explained in `DEMO_WALKTHROUGH.md` | `--pick` output, page tile |
| `results/f2_economics/fig_equity_curve.png`, `fig_cost_grid.png` | Honest: losses shown, net and gross both drawn, REAL DATA stated. Title and the "STT 0.02% to 0.05%" note are typed, not read from `TEST_START`, `TEST_END`, `STT_SCHEDULE` (they agree today) | Low | OPEN (agree today; risk only if the constants change) | `experiments/f2_economics.py` lines 61, 73, 432, 434 |
| `results/connectedness_gamma/fig_connectedness_gamma.png` | Left panel y axis runs 70 to 90 on a 0 to 200 scale (stated in the label); the primary p 0.044 is unadjusted and the figure does not show that | Low | OPEN | `results/connectedness_gamma/fig_connectedness_gamma.png` |
| `results/pinning/fig_by_half_year.png`, `intraday_gamma/fig1`, `event_study/fig1`, `power_real/power_curves.png`, `p117 fig_placebo`, `replication_nse/fig_rolling_connectedness.png` | Viewed: REAL DATA or "simulations calibrated on real data" stated, intervals shown, small-sample bins labelled (n = 4 to 6 expiry days) | none | VERIFIED by eye | viewed directly from `results/` (not copied) |

## Not done

I did not view all 96 PNGs (cost). Unseen study figures: `p114_*`, `dealer_sign/*`, `pinning_mechanism`, `synthetic_control`, `event_study/fig2` and `fig3`, `pipeline_synthetic/*`. The deck's other 12 slides are text and tables only; I checked layout, not wording.
