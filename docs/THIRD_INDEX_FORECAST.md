# Does the Nifty forecasting hybrid also beat mcsGARCH on a third index, Fin Nifty? (P104 results)

**Verdict: REPLICATED.** The hybrid that was designed on Nifty, run on Fin Nifty (NSE's Nifty Financial Services
index) with nothing changed, has a lower forecast loss than the anchor paper's mcsGARCH at both registered horizons:
4.9% lower for the next 15 minutes and 6.1% lower for the next 30 minutes, on all 604 test days, with Holm-adjusted
p-values far below 0.05 (both 4.4e-12). On the same dates the gains are close to Nifty's (5.1% and 6.6%) and a little
larger than Bank Nifty's (4.2% and 5.3%). The three secondary results are mixed: the last hour (H2) and Fin Nifty's
expiry days (H3) are **not supported**, and the prediction that dealer-gamma (GEX) features do not help (H4) **held**.

**REAL DATA.** Every number below comes from real Indian market data: third-party 1-minute Fin Nifty bars (checked
day by day against NSE's official daily values), NSE's F&O bhavcopies (Fin Nifty options and futures), NSE's
participant-wise open interest file and India VIX. Nothing is simulated. Plan: `docs/prereg/P104_THIRD_INDEX_FORECAST.md`,
committed (a1e84a5, "P104 registered plan") before any Fin Nifty forecast or loss was computed. Outputs:
`results/pipeline_nse_finnifty/`. Run on 3 November 2026. The study (data search, plan, code, run and this write-up)
was done by an AI agent (Claude) working for Shreyan Thanki.

## 1. The answer in plain words

On Nifty, we found that the anchor paper's intraday volatility model (mcsGARCH), corrected by a machine-learning
model (LightGBM), forecasts the next 15 and 30 minutes of volatility better than mcsGARCH alone. Every choice in that
"hybrid" was made while looking at Nifty. P91 ran it unchanged on Bank Nifty and it won there too. This study wrote
down the same test in advance and ran exactly the same hybrid, with the same settings, on a third index.

* **Which third index.** Fin Nifty was the only candidate with free 1-minute index bars for 2023 to 2026 that pass
  the same quality check as the Nifty and Bank Nifty bars. Midcap Select was rejected (no index minute bars for those
  years could be found), and Sensex is not eligible because its options trade on BSE, whose files cannot be reached
  from this machine (plan, feasibility section).
* **The main test passed.** At 15 and 30 minutes the hybrid beat mcsGARCH on Fin Nifty clearly, in every regime,
  on non-expiry days, in the "development" and "hold-out" periods and after removing each model's worst 1% of
  windows. So the forecasting result is now seen on three NSE indices, each tested under a plan written in advance.
* **Last hour (H2): not supported.** In the 14:30 to 15:30 window the hybrid was only 1.0% better (p 0.30). On
  Nifty and Bank Nifty, over the same dates, it was 5.4% and 8.2% better. On Fin Nifty the last-hour gain appears
  only after November 2024 (5.0% better, p 0.030) and not before (2.5% worse).
* **Fin Nifty expiry days (H3): not supported.** On the 82 expiry days the hybrid was better at 15 and 30 minutes
  (3.9% and 5.3%), but with Holm-adjusted p-values of 0.093, above the 0.05 bar. The direction is the same as on the
  other two indices; 82 days were not enough to show it here.
* **Dealer gamma (H4): our prediction held.** Adding option-positioning (GEX) features did not improve the hybrid
  under any sign rule at any horizon (smallest Holm p 0.40). At 15 and 30 minutes GEX made it slightly worse. The
  Bank Nifty finding that GEX helped the 09:15 forecast did **not** reappear on Fin Nifty.

Section 9 lists why this is a weaker test than it looks: Fin Nifty moves with Bank Nifty and Nifty, it borrows
Nifty's VIX, its option features mean something different after November 2024, its minute bars come from a source
of unknown origin, and the project had used Fin Nifty data before.

## 2. How to read the numbers

* **What is forecast.** At a given minute, how much Fin Nifty will move over the next 15 minutes, the next 30
  minutes, or the last hour (14:30 to 15:30). "How much it moves" is the realized variance: the sum of squared
  1-minute log returns in that window. Forecasts are made every 15 minutes from 09:15 (25 a day at 15 minutes, 24
  at 30 minutes) and once a day for the last hour.
* **QLIKE** is the loss (lower is better): for a realized value y and a forecast f, y/f - log(y/f) - 1. Tables give
  each model's mean QLIKE **relative to mcsGARCH, in percent**: -4.9% means 4.9% lower (better) than mcsGARCH.
* **p-values** come from a one-sided Diebold-Mariano test on daily average losses (Newey-West variance,
  Harvey-Leybourne-Newbold correction, Student t), exactly as for Nifty and Bank Nifty: a small p means the model's
  lower loss is unlikely to be luck. "<0.001" means below 0.001; numbers like 4.4e-12 mean 4.4 times 10 to the
  power -12.
* **Holm adjustment** raises p-values when several tests are made together, so that the chance of even one false
  alarm stays at 5%.
* **Walk-forward.** The first 150 days (2 January to 9 August 2023) are only for training. Then every model,
  mcsGARCH included, is refitted on all earlier days before each block of 21 trading days and forecasts that block.
  Test days: 604, from 10 August 2023 to 27 January 2026 (the last day of the Fin Nifty bars), in 29 blocks.
* **Model names** are those of `docs/MODEL_ITERATIONS_REAL.md`. The registered model is `hybrid_base` ("hybrid, new
  base"): mcsGARCH's forecast times a LightGBM correction, no GEX.

## 3. The primary test, H1 (REAL DATA)

| Horizon | Test days | mcsGARCH mean QLIKE | hybrid mean QLIKE | Change | DM statistic | raw p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| next 15 minutes | 604 | 0.2554 | 0.2429 | -4.9% | 7.01 | 3.3e-12 | 4.4e-12 |
| next 30 minutes | 604 | 0.2036 | 0.1912 | -6.1% | 7.07 | 2.2e-12 | 4.4e-12 |

Both Holm-adjusted p-values are at most 0.05, so by the registered rule (plan Section 5) the result is
**replicated**. Source: rows `all`, `qlike`, `mcsgarch vs hybrid_base` of `tests_next15.csv` and `tests_next30.csv`;
Holm in `p104/hypotheses.csv`. The plan's power section expected DM statistics of about 5 to 7 if Fin Nifty behaved
like the first two indices; the observed 7.0 and 7.1 are at the top of that range.

The seed check agrees: rerun with three single LightGBM seeds at 30 minutes, the hybrid is 5.7% to 6.2% better than
mcsGARCH (the registered 5-seed average: 6.1%); the largest gap between two of its seeds is 0.56 points
(`seed_noise_next30.csv`). Seed noise is far smaller than the gain.

## 4. Secondary hypotheses (REAL DATA, each reported whatever its direction)

### 4.1 H2, the last hour: not supported

| Window | Test days | mcsGARCH mean QLIKE | hybrid mean QLIKE | Change | DM statistic | p |
| --- | --- | --- | --- | --- | --- | --- |
| 14:30 to 15:30 | 604 | 0.1471 | 0.1457 | -1.0% | 0.52 | 0.30 |

The hybrid is better, but by too little to tell from luck (p 0.30 against the 0.05 bar). The breakdowns (Section 6)
show where this comes from: in the weekly-options period (to 19 November 2024) the hybrid was 2.5% *worse* in the
last hour, and in the monthly-only period after it 5.0% better (p 0.030; this split carries no hypothesis). No model
of the 31 beat mcsGARCH significantly in Fin Nifty's last hour under QLIKE (smallest p 0.110, Section 5).

### 4.2 H3, Fin Nifty expiry days: not supported

| Horizon | Expiry days | mcsGARCH | hybrid | Change | DM statistic | raw p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| next 15 minutes | 82 | 0.3707 | 0.3563 | -3.9% | 1.53 | 0.065 | 0.093 |
| next 30 minutes | 82 | 0.3098 | 0.2935 | -5.3% | 1.70 | 0.046 | 0.093 |
| last hour (no hypothesis) | 82 | 0.2006 | 0.2093 | **+4.4%** (worse) | -0.86 | 0.80 | |

Both Holm p are above 0.05, so H3 is **not supported**. The direction at 15 and 30 minutes is the same as on Nifty
and Bank Nifty, and the size is in between, but Fin Nifty's expiry days are noisy (mcsGARCH's mean QLIKE is 0.37 on
them against 0.26 on all days) and there are only 82. As on Bank Nifty, the last hour of expiry days is the one place
where mcsGARCH had the lower loss, not significantly (the p for "mcsGARCH is better" is 0.20).

### 4.3 H4, the GEX ablation: prediction held

The test: the same hybrid trained once without and once with each set of gamma-exposure (GEX) features. Negative
change = adding GEX lowered the loss. The prediction was that none of the 12 Holm-adjusted p-values (4 sets x 3
targets, Holm over the 4 sets within each target) would be at most 0.05. None was.

| Target, all 604 days | gross (no sign) | US convention | participant (Pro) | client mirror | placebo (wrong days) |
| --- | --- | --- | --- | --- | --- |
| next 15 minutes | +0.13% (p 0.78, Holm 1.00) | +0.24% (0.91, Holm 1.00) | +0.29% (0.94, Holm 1.00) | +0.24% (0.90, Holm 1.00) | +0.18% (0.91) |
| next 30 minutes | +0.58% (0.98, Holm 1.00) | +0.32% (0.83, Holm 1.00) | +0.74% (0.99, Holm 1.00) | +0.83% (1.00, Holm 1.00) | +0.34% (0.98) |
| last hour | -1.08% (0.10, Holm 0.40) | -1.00% (0.13, Holm 0.40) | -0.45% (0.30, Holm 0.47) | -0.65% (0.24, Holm 0.47) | -0.19% (0.31) |

At 15 and 30 minutes every GEX set made the hybrid slightly worse, as extra columns without information tend to do.
In the last hour all four sets lowered the loss by 0.4% to 1.1%, none significantly. Against the placebo set (the
same columns as the US set, taken from other days), the real US set was 0.07% worse at 15 minutes (p 0.66), 0.01%
better at 30 minutes (p 0.48) and 0.81% better in the last hour (p 0.17): no sign of information in the real
features. The verdict is **"prediction held"**, in the sense of the plan: no improvement was shown.

On expiry days (reported without a verdict) no set had a Holm p below 0.29 at any horizon (smallest: gross, last
hour, 2.5% better, raw p 0.072).

**The Bank Nifty 09:15 effect (exploratory).** On Bank Nifty all four GEX sets helped the 09:15 forecast at 15
minutes by 7.5% to 8.6%. On Fin Nifty the same cells (`ablation_next15.csv`, group `open_0915`) are much smaller and
not significant: gross 1.2% better (p 0.072, Holm within the anchor group 0.29), participant 1.1% (0.098, Holm 0.29),
US 0.3% and client 0.5%; placebo 0.2% worse. At later anchors every set was 0.2% to 0.4% worse. At 30 minutes and
09:15 the changes ranged from 0.1% better to 1.5% worse. So the Bank Nifty 09:15 gain did not repeat on a third
index.

## 5. Every model against mcsGARCH (REAL DATA)

Cells: mean QLIKE relative to mcsGARCH, and in brackets the one-sided DM p that the model beats mcsGARCH. 604 test
days, 82 of them Fin Nifty expiry days. The bold row is the registered model.

| Model | 15 min, all days | 15 min, expiry days | 30 min, all days | 30 min, expiry days | last hour, all days | last hour, expiry days |
| --- | --- | --- | --- | --- | --- | --- |
| naive (last window) | +169.9% (1.000) | +152.2% (1.000) | +91.3% (1.000) | +89.8% (1.000) | +132.3% (1.000) | +272.6% (0.999) |
| naive (same window yesterday) | +243.8% (1.000) | +354.9% (0.994) | +217.8% (1.000) | +330.3% (0.992) | +190.6% (1.000) | +337.1% (0.998) |
| mcsGARCH refitted every 63 days | -0.0% (0.450) | -0.6% (0.243) | +0.4% (0.744) | +0.8% (0.721) | +2.5% (0.986) | +5.5% (0.884) |
| HAR | +6.5% (1.000) | +10.0% (0.922) | +6.1% (0.999) | +11.7% (0.882) | -3.2% (0.248) | +8.3% (0.696) |
| VIX | +69.7% (1.000) | +119.8% (0.978) | +79.4% (1.000) | +134.0% (0.971) | +87.7% (1.000) | +82.9% (0.953) |
| implied (straddle) | +48.2% (1.000) | +58.9% (1.000) | +52.1% (1.000) | +61.9% (1.000) | +69.5% (1.000) | +34.2% (0.828) |
| mcsGARCH, implied daily level | -2.5% (0.014) | -5.8% (0.054) | -2.7% (0.053) | -6.8% (0.088) | +3.2% (0.722) | -17.1% (0.229) |
| ridge log-HAR, v1 base | -0.3% (0.336) | +0.4% (0.561) | -1.9% (0.052) | -1.9% (0.325) | +3.1% (0.865) | +7.2% (0.831) |
| ridge log-HAR, new base | -1.9% (0.013) | -2.8% (0.041) | -3.7% (<0.001) | -3.8% (0.076) | -1.1% (0.383) | -4.6% (0.383) |
| LightGBM, v1 base | -2.1% (0.252) | +10.5% (0.744) | -1.9% (0.344) | +17.0% (0.757) | +0.2% (0.518) | +27.9% (0.904) |
| LightGBM, new base | -2.4% (0.207) | +8.8% (0.726) | -2.4% (0.304) | +15.2% (0.745) | -0.2% (0.484) | +14.3% (0.698) |
| LightGBM tuned, new base | -2.0% (0.249) | +9.4% (0.744) | -1.6% (0.366) | +17.3% (0.769) | -2.2% (0.314) | +2.4% (0.544) |
| hybrid, v1 base | -4.7% (<0.001) | -4.6% (0.022) | -6.0% (<0.001) | -6.2% (0.027) | -0.5% (0.400) | +9.2% (0.963) |
| hybrid, v1 base + v1 GEX | -4.8% (<0.001) | -3.9% (0.019) | -5.5% (<0.001) | -5.2% (0.012) | -1.5% (0.171) | +8.7% (0.967) |
| **hybrid, new base** | -4.9% (<0.001) | -3.9% (0.065) | -6.1% (<0.001) | -5.3% (0.046) | -1.0% (0.301) | +4.4% (0.804) |
| hybridiv, new base | -5.1% (<0.001) | -7.2% (0.011) | -6.3% (<0.001) | -9.4% (0.025) | -4.5% (0.159) | -16.4% (0.212) |

The models with GEX (each base model plus one GEX set):

| Model | 15 min, all days | 15 min, expiry days | 30 min, all days | 30 min, expiry days | last hour, all days | last hour, expiry days |
| --- | --- | --- | --- | --- | --- | --- |
| `linear_gex_gross` | -1.9% (0.019) | -3.3% (0.011) | -2.6% (0.019) | -3.6% (0.074) | +0.0% (0.501) | +0.2% (0.507) |
| `linear_gex_us` | -2.1% (0.009) | -2.8% (0.024) | -3.1% (0.006) | -2.8% (0.111) | -0.8% (0.416) | -2.8% (0.425) |
| `linear_gex_participant` | -1.7% (0.034) | -3.6% (0.041) | -3.1% (0.007) | -4.6% (0.083) | +2.2% (0.756) | +9.5% (0.892) |
| `linear_gex_client` | -1.6% (0.031) | -2.1% (0.095) | -2.4% (0.031) | -1.9% (0.233) | +2.3% (0.774) | +11.4% (0.920) |
| `linear_gex_placebo` | -1.7% (0.020) | -2.4% (0.057) | -3.6% (<0.001) | -4.6% (0.027) | -1.3% (0.368) | -5.6% (0.364) |
| `hybrid_gex_gross` | -4.8% (<0.001) | -3.4% (0.083) | -5.5% (<0.001) | -4.5% (0.053) | -2.0% (0.111) | +1.8% (0.641) |
| `hybrid_gex_us` | -4.7% (<0.001) | -3.2% (0.103) | -5.8% (<0.001) | -4.4% (0.050) | -1.9% (0.110) | +3.8% (0.817) |
| `hybrid_gex_participant` | -4.6% (<0.001) | -3.5% (0.065) | -5.4% (<0.001) | -4.6% (0.043) | -1.4% (0.172) | +4.2% (0.833) |
| `hybrid_gex_client` | -4.7% (<0.001) | -2.9% (0.152) | -5.3% (<0.001) | -4.2% (0.058) | -1.6% (0.157) | +2.5% (0.704) |
| `hybrid_gex_placebo` | -4.7% (<0.001) | -3.6% (0.075) | -5.8% (<0.001) | -4.5% (0.065) | -1.1% (0.259) | +4.3% (0.782) |
| `hybridiv_gex_gross` | -5.1% (<0.001) | -7.0% (0.015) | -6.1% (<0.001) | -9.4% (0.026) | -4.0% (0.194) | -18.4% (0.182) |
| `hybridiv_gex_us` | -4.9% (<0.001) | -6.7% (0.030) | -6.3% (<0.001) | -9.1% (0.037) | -4.0% (0.196) | -16.6% (0.216) |
| `hybridiv_gex_participant` | -5.0% (<0.001) | -7.3% (0.018) | -5.9% (<0.001) | -9.2% (0.030) | -3.5% (0.216) | -18.6% (0.176) |
| `hybridiv_gex_client` | -5.0% (<0.001) | -7.4% (0.017) | -6.3% (<0.001) | -9.7% (0.024) | -3.8% (0.198) | -18.0% (0.181) |
| `hybridiv_gex_placebo` | -5.0% (<0.001) | -7.6% (0.015) | -6.2% (<0.001) | -9.7% (0.024) | -4.1% (0.184) | -16.0% (0.217) |

What stands out (descriptive, no hypothesis):

* The picture at 15 and 30 minutes is the same as on Nifty and Bank Nifty: simple tools (HAR, VIX, the straddle
  alone, the 63-day refit) do not beat mcsGARCH, and every hybrid does.
* **LightGBM without mcsGARCH is weak on Fin Nifty**: 2.4% better on all days (not significant) and 9% to 15%
  *worse* on expiry days. What wins is the combination: mcsGARCH's intraday shape, corrected by LightGBM.
* **The registered hybrid is again not the best model.** `hybridiv` (the same correction on top of an mcsGARCH whose
  daily level comes from option prices) is about equal at 15 and 30 minutes (5.1% and 6.3% against 4.9% and 6.1%)
  and better in the last hour (4.5% against 1.0%, neither significant) and on expiry days (7.2% and 9.4%, raw p
  0.011 and 0.025). The same happened on Bank Nifty. This was not a hypothesis and does not change any verdict.
* **Last hour:** no model, with or without GEX, beats mcsGARCH significantly on all days (smallest p 0.110).

## 6. Breakdowns (REAL DATA, no hypotheses)

### 6.1 By expiry regime, and weekly against monthly-only options

The pipeline's three regimes (cut at 20 November 2024, when Fin Nifty's weekly options were gone, and 1 September
2025, when its monthly expiry moved back to Tuesday). Hybrid against mcsGARCH:

| Period | Test days | 15 min | 30 min | last hour |
| --- | --- | --- | --- | --- |
| weekly options (to 19 Nov 2024) | 312 | -3.9% (<0.001) | -4.5% (<0.001) | **+2.5%** (0.84) |
| monthly only, 20 Nov 2024 to 31 Aug 2025 | 192 | -6.5% (<0.001) | -8.8% (<0.001) | -2.8% (0.20) |
| monthly only, from 1 Sep 2025 | 100 | -5.1% (<0.001) | -6.4% (<0.001) | -10.1% (0.004) |
| **monthly only, pooled** (from 20 Nov 2024) | 292 | -6.1% (9e-13) | -8.2% (3e-13) | -5.0% (0.030) |

On expiry days only:

| Period | Expiry days | 15 min | 30 min | last hour |
| --- | --- | --- | --- | --- |
| weekly options (to 19 Nov 2024) | 67 | -2.5% (0.18) | -3.0% (0.17) | +1.5% (0.62) |
| monthly only (from 20 Nov 2024) | 15 | -11.9% (0.027) | -17.2% (0.029) | +32.1% (0.92) |

On the other (non-expiry) days of each regime the hybrid was 4.6%, 6.1% and 4.6% better at 15 minutes and 5.2%,
8.1% and 5.7% at 30 minutes (all p below 0.001); on all 522 non-expiry days 5.1% and 6.3%. So, as on Bank Nifty,
the gain is larger in the monthly-only period and does not depend on expiry days. The expiry-day gain at 15 and 30
minutes comes mostly from the 15 monthly-only expiry days (see caveat 4 in Section 9). (Sources:
`scoreboard_<target>.csv` and `tests_<target>.csv` for the regimes; `p104/period_split.csv` for the pooled split,
computed with the pipeline's `daily_losses` and `diebold_mariano`.)

### 6.2 The 09:15 forecast against later ones

| Model | 15 min, 09:15 only | 15 min, without 09:15 | 30 min, 09:15 only | 30 min, without 09:15 |
| --- | --- | --- | --- | --- |
| HAR | +45.8% (1.000) | +3.3% (0.986) | +43.3% (1.000) | +2.9% (0.908) |
| LightGBM, new base | -14.0% (<0.001) | -1.5% (0.317) | -21.0% (<0.001) | -0.8% (0.439) |
| **hybrid, new base** | -10.8% (0.016) | -4.4% (<0.001) | -17.0% (<0.001) | -5.2% (<0.001) |
| hybridiv, new base | -18.2% (0.035) | -4.0% (<0.001) | -26.3% (0.033) | -4.6% (0.001) |

As on Nifty and Bank Nifty, the gain is largest at the open, but most of the evidence comes from the later anchors:
without 09:15 the hybrid is still 4.4% and 5.2% better with p below 0.001. LightGBM alone wins only at 09:15.
(`gain_by_anchor_<target>.csv` has every anchor minute.)

### 6.3 Trimmed QLIKE, MSE and the Model Confidence Set

| Model | trimmed QLIKE 15 min | trimmed 30 min | trimmed last hour | MSE 15 min (p) | MSE 30 min (p) | MSE last hour (p) |
| --- | --- | --- | --- | --- | --- | --- |
| HAR | +5.2% | +5.0% | -3.6% | -99.8% (0.158) | -99.7% (0.158) | -50.6% (0.128) |
| mcsGARCH, implied daily level | -1.2% | -0.5% | +10.5% | -99.8% (0.158) | -99.8% (0.158) | -37.7% (0.189) |
| ridge log-HAR, new base | -1.1% | -2.5% | +4.0% | -99.8% (0.158) | -99.7% (0.158) | -60.8% (0.119) |
| LightGBM, new base | -4.7% | -6.1% | -1.4% | -99.8% (0.158) | -99.7% (0.158) | -35.6% (0.147) |
| **hybrid, new base** | -5.5% | -6.4% | -2.6% | -43.5% (0.158) | -26.9% (0.158) | -16.8% (0.071) |
| hybridiv, new base | -4.7% | -5.2% | -0.4% | -99.8% (0.158) | -99.8% (0.158) | -53.7% (0.136) |

* **Trimmed QLIKE** (each model's own largest 1% of losses removed): the hybrid stays 5.5%, 6.4% and 2.6% better,
  so the win is not driven by a few extreme windows.
* **MSE** (squared error) on Fin Nifty is dominated by **one forecast**: mcsGARCH's 09:15 forecast of 5 June 2024
  (Section 6.6). It holds 98% and 96% of mcsGARCH's total squared error at 15 and 30 minutes, which is why almost
  every other model shows "-99.8%" and why no MSE test can be significant (p 0.158 for one dominant day). The hybrid,
  which multiplies mcsGARCH's forecast, inherits most of that error. Of all 31 models, none beats mcsGARCH under MSE
  at 15 or 30 minutes; in the last hour 6 do (`mcsgarch_r63`, `hybrid_v1` and the four `hybrid_gex_*`, p 0.021 to
  0.048). Without 5 June 2024 the hybrid's MSE is 4.5% lower at both 15 and 30 minutes (p 0.058 and 0.065) and 18.0%
  lower in the last hour (p 0.033) (exploratory, Section 6.6). As on Nifty and Bank Nifty, MSE is very noisy.
* **Model Confidence Set** (models that cannot be separated from the best at the 10% level, `mcs_*.csv`): under
  QLIKE on all days, `hybrid_base` is in the set at all three horizons; mcsGARCH is out at 15 and 30 minutes and in
  for the last hour (sets of 17, 23 and 27 of 32 models). On expiry days both are in at every horizon (sets of 30,
  29 and 29). Under MSE both are in everywhere.

### 6.4 "Development" and "hold-out" periods

For Fin Nifty all test days are out of sample for the design, so this split (at 1 July 2025) has no special
meaning; it is reported because the pipeline prints it.

| Model | 15 min, to 30 Jun 2025 (462 days) | 15 min, from 1 Jul 2025 (142) | 30 min, to 30 Jun 2025 | 30 min, from 1 Jul 2025 | last hour, to 30 Jun 2025 | last hour, from 1 Jul 2025 |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM, new base | -1.7% (0.323) | -5.9% (<0.001) | -1.1% (0.423) | -9.2% (<0.001) | +3.1% (0.684) | -12.4% (0.001) |
| LightGBM tuned, new base | -1.0% (0.386) | -6.3% (<0.001) | -0.0% (0.500) | -10.1% (<0.001) | +0.1% (0.510) | -10.8% (0.004) |
| **hybrid, new base** | -4.8% (<0.001) | -5.5% (<0.001) | -5.9% (<0.001) | -7.3% (<0.001) | +1.4% (0.749) | -9.7% (<0.001) |
| hybridiv, new base | -4.9% (<0.001) | -6.2% (<0.001) | -5.7% (0.002) | -9.8% (<0.001) | -2.0% (0.362) | -14.0% (0.002) |

The last-hour gain appears only in the later period, which matches Section 6.1.

### 6.5 Seed noise (30 minutes)

Three single seeds, against mcsGARCH: `hybrid_base` -6.20%, -5.86%, -5.67%; `lgbm_base` -2.53%, -1.76%, -1.83%;
`hybrid_gex_participant` -5.26%, -5.21%, -5.18%; `hybridiv_base` -6.29%, -6.34%, -6.48%. The largest gap between two
seeds of one model is 0.79 points (`lgbm_base`, `seed_noise_next30.csv`). The registered forecasts average 5 seeds,
which shrinks this further.

### 6.6 The benchmark's forecast of 5 June 2024 (exploratory, not in the plan)

On 4 June 2024 (general-election results) Fin Nifty fell about 7% in the day. The next morning, mcsGARCH's 09:15
forecasts were **135 times** (15 minutes) and **140 times** (30 minutes) the variance that followed. The hybrid,
which multiplies mcsGARCH's forecast, was 101 and 119 times too high. The cause is mcsGARCH's daily level: the refit
in force (from 23 May 2024) used the paper's EGARCH model for the daily part, which reacts very strongly to a 7% day.
`mcsgarch_iv`, which has the same intraday parts but takes its daily level from option prices, was only 1.8 times
too high. The same forecast was 20 and 24 times too high on Nifty, and 5 and 6 times on Bank Nifty, whose refit at
that date had fallen back to EWMA (`p104/june5_2024.csv`).

Under QLIKE this one day hardly matters: without 5 June 2024 the H1 comparison is 4.8% (DM 6.53, p 7e-11) and 6.1%
(DM 6.84, p 1e-11), the last hour 0.9% (p 0.31). Under MSE it is everything (Section 6.3). The numbers are in
`p104/exploratory.csv`, written by `experiments/p104_finnifty_exploratory.py` after the registered results were seen;
the headline keeps all days, as the plan says.

## 7. Nifty and Bank Nifty on the same dates (REAL DATA)

The Nifty v2 forecasts (`results/pipeline_nse_v2/forecasts_<target>.csv`) and the Bank Nifty P91 forecasts
(`results/pipeline_nse_banknifty/forecasts_<target>.csv`), neither rerun, restricted to the same 604 test dates and
scored the same way. Each index uses its own expiry days. Each index's models were trained on its own walk-forward,
so this is a description, not a test of the differences. Cells: hybrid against mcsGARCH, DM statistic in brackets.

| Target | Nifty, all days | Bank Nifty, all days | Fin Nifty, all days | Nifty, its 130 expiry days | Bank Nifty, its 82 expiry days | Fin Nifty, its 82 expiry days |
| --- | --- | --- | --- | --- | --- | --- |
| next 15 minutes | -5.1% (9.40) | -4.2% (8.06) | -4.9% (7.01) | -4.1% (4.05) | -2.9% (2.61) | -3.9% (1.53) |
| next 30 minutes | -6.6% (7.61) | -5.3% (7.27) | -6.1% (7.07) | -4.4% (2.84) | -3.5% (2.36) | -5.3% (1.70) |
| last hour | -5.4% (2.13) | -8.2% (4.84) | -1.0% (0.52) | -5.8% (1.60) | +4.7% (-0.74) | +4.4% (-0.86) |

At 15 and 30 minutes Fin Nifty's gain is 96% and 92% of Nifty's on the same dates and larger than Bank Nifty's. In the
last hour it is much smaller than on the other two. On expiry days the Fin Nifty gains are of similar size to the
others', but noisier (smaller DM statistics). Source: `p104/other_indices_same_dates.csv`.

## 8. Data checks (recomputed by the run; all match the pre-registration)

* 754 trading days, 2 January 2023 to 27 January 2026 (`daily_market.csv`, `run_info.json`). Training to 9 August
  2023; 604 test days from 10 August 2023; 25, 24 and 1 forecasts a day for the three targets.
* No day without a Fin Nifty bhavcopy, without India VIX or without Fin Nifty futures rows.
* Test days by regime: 312, 192 and 100. Fin Nifty expiry days among them: 82 (30 monthly), 67 in the weekly period
  and 15 after it. Development 462 days, hold-out 142.
* Daily GEX panel: the Pro-rule gamma-flip level is missing on 468 of 754 days (as expected: no flip within 4% of the
  forward), the US-convention flip on 5 days and the chain's at-the-money volatility on 1 day. Implied panel: the
  next-expiry columns are missing on 11 days. All as registered (plan Section 2).
* The run's arguments (`run_info.json`) are exactly those of the plan's command. The stdout log
  (`results/pipeline_nse_finnifty_stdout.txt`) shows no warning or skipped step. The run took 61 minutes (3,680
  seconds) on one thread.
* **mcsGARCH's EWMA fallback.** When the EGARCH fit of mcsGARCH's daily level fails, the code (unchanged from Nifty)
  falls back to a simple EWMA variance. On Fin Nifty this happened in 3 of 29 refits (the first three, which forecast
  the 63 test days from 10 August to 10 November 2023), against 2 of 38 on Nifty and 8 of 32 on Bank Nifty
  (`mcsgarch_params.csv`, column `daily_method`). It did not flatter the hybrid: on those 63 days the hybrid did less
  well (3.0% and 3.8% better at 15 and 30 minutes, 1.0% worse in the last hour) than on the other 541 days (5.1%, 6.3%
  and 1.3% better). (Exploratory, `p104/exploratory.csv`.)

## 9. Fairness caveats: what this result does and does not show

1. **Fin Nifty is not independent of Nifty or Bank Nifty.** Its 604 test dates are a subset of both other samples,
   with the same news, the same India VIX and the same participant file, and about a third of Fin Nifty is the same
   big banks that make up Bank Nifty. A success shows that the design is not specific to one index's data. It is
   not a new sample in time.
2. **The minute bars are the weakest of the three and of unknown origin.** The bars (GitHub repository
   `ayush-roy-21/EigenTrade`, pinned commit) have the day's high and low within 5 basis points of NSE's on 98.9% and
   99.2% of days (Bank Nifty 99.4% and 99.3%, Nifty 100% and 99.9%), and 50 bars matched an independent sample
   exactly. The repository states no licence and no source. Nothing checks every minute.
3. **India VIX stands in for a Fin Nifty VIX** (NSE publishes none). This should, if anything, make the hybrid weaker.
4. **The option features and mcsGARCH's expiry curve after November 2024.** Fin Nifty's last weekly options expired
   on 19 November 2024. From then on the "nearest expiry" contract that feeds GEX and the near-expiry implied
   features is a monthly up to 25 trading days away, and mcsGARCH applies an expiry-day curve learnt mostly from
   weekly expiries to monthly ones. The hybrid's calendar features can repair that curve, which would be an easy win
   rather than new information. The large gains on the 15 monthly-only expiry days (11.9% and 17.2%) may partly be
   this repair. The main result does not come from there: on the 522 non-expiry days the hybrid was 5.1% and 6.3%
   better.
5. **The project was not blind to Fin Nifty.** Earlier studies used Fin Nifty's expiry calendar and a daily
   volatility measure (listed in the plan), although none made an intraday forecast and none shaped the hybrid. The
   Bank Nifty results were known when the plan was written; nothing was changed in response.
6. **A shorter sample with no closing-auction days** (604 test days, against 660 for Bank Nifty and 792 for Nifty).
7. **The benchmark is the paper's model as coded here, with its quirks**: an EWMA fallback in 3 of 29 refits and one
   very large forecast on 5 June 2024 (Sections 6.6 and 8). Neither drives the QLIKE result.

## 10. Deviations from the pre-registration

**None in the registered analysis.** The command of plan Section 1 was run unchanged, nothing crashed, and no model,
feature, setting, sample, test or threshold was changed. For completeness:

* **How the code was loaded.** As for P91, the package is installed in editable mode from the main checkout, so the
  command was run from a separate git worktree at commit a1e84a5 (the plan's commit) with `PYTHONPATH` set to that
  worktree's `src`, with `OMP_NUM_THREADS=1`. While the run was going, the worktree was moved onto newer commits of
  the branch (other studies' files). `git diff a1e84a5` shows no change under `src/`, in `scripts/run_pipeline.py` or
  in `experiments/p104_finnifty_summary.py`, so the code that ran is the registered code. This changes no number.
* **One helper script was added after the results were seen**, `experiments/p104_finnifty_exploratory.py`. It fits
  nothing; it writes the two exploratory looks of Sections 6.6 and 8 (the EWMA-fallback split and the comparison
  without 5 June 2024) with the pipeline's own functions.
* **Labelled exploratory looks** that were not in the plan, reported because they help to read the result: the GEX
  ablation at the 09:15 anchor (Section 4.3), the regime split into expiry and other days (Section 6.1), the 5 June
  2024 forecast and the comparison without that day (Section 6.6), and the EWMA-fallback split (Section 8). None of
  them changes a verdict.

## 11. What this changes in the project's claims

* **The forecasting headline is now replicated on three indices under registered plans** (Nifty, Bank Nifty P91, Fin
  Nifty P104): at 15 and 30 minutes the hybrid beats mcsGARCH by 4.9% and 6.1% on Fin Nifty, 3.9% and 5.0% on Bank
  Nifty, and 5.8% and 7.6% on Nifty (each on its own full sample).
* **The last-hour gain is index-dependent.** It holds on Nifty and Bank Nifty but not on Fin Nifty (1.0%, p 0.30),
  where it appears only after November 2024. Claims about the last hour should say so.
* **Expiry days:** same direction on all three indices, significant on Nifty and Bank Nifty, not on Fin Nifty after
  Holm (0.093 at both horizons, 82 days).
* **GEX:** the "no forecasting value" result holds on Fin Nifty, and the Bank Nifty exception (GEX helping the 09:15
  forecast) did not repeat. Across the three indices, GEX helped on one (Bank Nifty, at 15 minutes only).
* **MSE against mcsGARCH** should not be read as evidence either way: on all three indices a single forecast after
  the 4 June 2024 crash holds a large share of mcsGARCH's squared error (98% on Fin Nifty, 79% on Nifty, 42% on Bank
  Nifty, at 15 minutes).

## 12. Files and how to reproduce

* Plan: `docs/prereg/P104_THIRD_INDEX_FORECAST.md` (with the data search and the bar check).
* Data: `scripts/fetch_finnifty_data.py` downloads the bars (pinned commit and SHA-256), the official-day file and
  the bhavcopy folder into `data/raw/` (not in git; listed in `data/MANIFEST.csv`). Bar check:
  `results/data_validation_finnifty/`.
* Pipeline outputs (REAL DATA): `results/pipeline_nse_finnifty/` (forecasts, scoreboards, DM tests, ablation tables,
  Model Confidence Sets, mcsGARCH parameters, seed noise, figures, `run_info.json`); log
  `results/pipeline_nse_finnifty_stdout.txt`.
* Registered summaries: `results/pipeline_nse_finnifty/p104/` (`hypotheses.csv`, `h4_ablation_hybrid.csv`,
  `scoreboard_vs_mcsgarch.csv`, `period_split.csv`, `other_indices_same_dates.csv`, `mcs_membership.csv`,
  `data_checks.json`). Exploratory: `p104/exploratory.csv` and `p104/june5_2024.csv`.
* Reproduce, from the repository root:

```
python scripts/fetch_finnifty_data.py --raw data/raw
OMP_NUM_THREADS=1 python scripts/run_pipeline.py --source nse --symbol FINNIFTY --end 2026-01-27 \
    --n-seeds 5 --seed-study 3 --verbose --out results/pipeline_nse_finnifty \
    > results/pipeline_nse_finnifty_stdout.txt 2>&1
python experiments/p104_finnifty_summary.py
python experiments/p104_finnifty_exploratory.py
```

The same steps are in `scripts/reproduce_all.py` (optional steps `validate_finnifty_bars`, `finnifty_pipeline`,
`p104_finnifty_summary` and `p104_finnifty_exploratory`).
