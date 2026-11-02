# DRY RUN on past data, 2025-01-01 to 2026-07-31: software check only

> **This is NOT the forward test.** Every day in this window was already used to find and study the effects (REAL_DATA_RESULTS R2 and R5, FALSIFICATION T4), so nothing here is evidence for or against H1 or H2. It only shows that `scripts/forward_test.py` runs end to end, reproduces published numbers and applies the pre-registered decision rules. The date caps do not apply to past data: each test uses the whole window unless its count rule (130 negative-GEX days, 100 weeks) is reached inside it.

Mode: dry run on past data. Code checked out: 0ca4e5242957da7d8228294975b8f640f96885d3. Inputs: built from data/raw with scripts/run_pipeline.py load_market and niftygex.features.build_gex_panel.

## Verdicts (pre-registered decision rules)

| Hypothesis | Stopping point | Sample | Estimate | One-sided p, predicted direction | Holm p | One-sided p, opposite direction | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | 2026-07-31 (window end (dry run: past data, the count rule was not reached inside the window)) | 389 days (90 negative GEX) | r_rod_x_neg -0.07965 (HAC SE 0.02949) | 0.003455 | 0.003455 | 0.9965 | DRY RUN: confirm |
| H2 | 2026-07-31 (window end (dry run: past data, the count rule was not reached inside the window)) | 48 weeks | mean 0.3425 log points (40.8% more variance) | 3.167e-05 | 6.334e-05 | 1 | DRY RUN: confirm |

Confirm: Holm p at most 0.05. Contradict: opposite-direction p below 0.05. Otherwise fail to confirm. Not evaluable: H1 with fewer than 30 negative-GEX or 30 other days, H2 with fewer than 20 weeks.

## Secondary tests (not in the Holm family; cannot change a verdict)

| Test | Sample | Estimate | p, predicted | p, opposite | Reading |
| --- | --- | --- | --- | --- | --- |
| H1b: r_last 14:59 to 15:15 (stops at the freeze) | 389 days | r_rod_x_neg -0.03935 (SE 0.02203) | 0.03699 | 0.963 | supports (secondary, unadjusted; cannot change a verdict) |
| H2b: Sensex expiry days out of the baseline | 46 weeks | mean 0.4827 (62%) | 3.066e-06 | 1 | supports (secondary, unadjusted; cannot change a verdict) |

## Descriptive only (no test, no decision)

* H1 sample, momentum only: slope of r_last on r_rod -0.01948 (HAC SE 0.01651), 389 days.
* H2 (14:15 to 15:15): 40.8% more variance on the expiry day, bootstrap 95% interval 20.7% to 63.3%; expiry day higher in 72.9% of weeks; t-test of the mean, one-sided p 2.839e-05; two-sided signed-rank p 6.334e-05.
* H2 weeks, R2's original last hour (bars 14:30 to 15:29; on auction days it includes the frozen minutes and the auction jump): 42.2% (48 weeks), two-sided signed-rank p 3.766e-05.

## Data and software checks

* minute_bars_vs_nse_official: {"checked": true, "days_compared": 389, "days_without_official_values": 0, "high_or_low_off_by_more_than_5bp": 0, "open_off_by_more_than_5bp": 0, "share_bad": 0.0, "bad_days": [], "passes": true}
* closing_auction_structure: {"auction_days_in_window": 0}
* r_rod_and_r_last_equal_momentum_gex_daily_returns_max_abs_diff_bp: 0.0
* rv_last_hour_rebuilt_from_bars_vs_pipeline_max_rel_diff: 3.658406910744816e-12
* lead_in_trading_days_before_start: 30
* dry_run_reproduces_published_numbers: {"applies": true, "h1_vs_falsification_t4_pre_auction": {"published_coef": -0.0796468723181795, "here": -0.0796468723181795, "published_p_one_sided": 0.0034547415745977, "here_p": 0.003454741574597592, "published_days": 389, "here_days": 389, "published_negative_days": 90, "here_negative_days": 90, "match": true}, "h2_original_last_hour_vs_closing_auction_r2_split": {"published_pct": 42.18871345797499, "here_pct": 42.18871345797499, "published_p_two_sided": 3.7655198283914615e-05, "here_p_two_sided": 3.7655198283914615e-05, "published_weeks": 48, "here_weeks": 48, "match": true}}
* rebuilt_panel_vs_committed_run: {"committed_run": "results/pipeline_nse_v2", "overlapping_days": 419, "max_rel_diff_net_gex_participant": 2.220446049250313e-16, "sign_mismatches_net_gex_participant": 0, "expiry_label_mismatches": 0}

Files: `summary.json` (everything above, plus code and data hashes), `h1_days.csv`, `h2_weeks.csv`.
