# Does the Nifty forecasting hybrid also beat mcsGARCH on Bank Nifty? (P91 results)

**Verdict: REPLICATED.** The hybrid that was designed on Nifty, run on Bank Nifty with nothing changed, has a
lower forecast loss than the anchor paper's mcsGARCH at both registered horizons: 3.9% lower for the next 15
minutes and 5.0% lower for the next 30 minutes, on all 660 test days, with Holm-adjusted p-values far below
0.05 (5e-12 and 2e-11). The gains are smaller than on Nifty over the same dates (5.5% and 6.9%).

*Note added 3 November 2026 after study P116:* here "mcsGARCH" and "the anchor paper's mcsGARCH" (or model) mean our
implementation (a separate time-of-day curve for expiry days, a 5-minute smoothing of the curves, a zero mean), not
the paper's own specification. Against the paper's specification (one curve, no smoothing, ARMA(1,1) mean) see
`docs/P116_BENCHMARK_AS_PRINTED.md`; there the hybrid's Bank Nifty gain is 4.73% and 6.20% at 15 and 30 minutes,
larger than the 3.9% and 5.0% in this document, because on Bank Nifty our implementation forecasts better than the
paper's specification (P116 Section 7).

**REAL DATA.** Every number below comes from real Indian market data: third-party 1-minute Bank Nifty bars
(checked day by day against NSE's official values), NSE's F&O bhavcopies (Bank Nifty options and futures), NSE's
participant-wise open interest file and India VIX. Nothing is simulated. Plan: `docs/prereg/P91_BANKNIFTY_FORECAST.md`,
committed (97eb0a1) before any Bank Nifty forecast or loss was computed. Outputs: `results/pipeline_nse_banknifty/`.
Run on 3 November 2026.

## 1. The answer in plain words

On Nifty, we found that the anchor paper's intraday volatility model (mcsGARCH), corrected by a machine-learning
model (LightGBM), forecasts the next 15 and 30 minutes of volatility better than mcsGARCH alone. Every choice in
that "hybrid" was made while looking at Nifty, so it might only fit Nifty's quirks. To check, we wrote down a test
in advance and then ran exactly the same hybrid, with the same settings, on Bank Nifty, NSE's other big index.

* **The main test passed.** At 15 and 30 minutes the hybrid beat mcsGARCH on Bank Nifty, clearly and on most
  sub-periods. So the Nifty result is not just a quirk of Nifty's data.
* **The gain is smaller than on Nifty** at 15 and 30 minutes (about 70% of Nifty's gain over the same dates),
  and larger in the last hour of the day (8.1% against 5.7%).
* **Last hour (H2): passed.** On all days the hybrid is 8.1% better in the 14:30 to 15:30 window.
* **Bank Nifty expiry days (H3): passed** at 15 and 30 minutes (2.8% and 3.4% better). In the last hour of the
  84 expiry days the hybrid was 3.0% *worse* than mcsGARCH (not significant, no hypothesis).
* **Dealer gamma (H4): our prediction failed.** We predicted, as on Nifty, that adding option-positioning (GEX)
  features would not help. On Bank Nifty they did help at 15 minutes, by 0.6% to 0.8% under every sign rule, and
  a placebo version of the same features did not. Looking closer (not planned), all of that help is in the 09:15
  forecast, the first window of the day, and none of it at later times. At 30 minutes and in the last hour GEX
  did not help.

Section 9 lists why this is a weaker test than it looks: Bank Nifty moves with Nifty, it borrows Nifty's VIX,
its option features mean something different after November 2024, and the project had used Bank Nifty data
before.

## 2. How to read the numbers

* **What is forecast.** At a given minute, how much Bank Nifty will move over the next 15 minutes, the next 30
  minutes, or the last hour (14:30 to 15:30). "How much it moves" is the realized variance: the sum of squared
  1-minute log returns in that window. Forecasts are made every 15 minutes from 09:15 (25 a day at 15 minutes,
  24 at 30 minutes) and once a day for the last hour.
* **QLIKE** is the loss (lower is better): for a realized value y and a forecast f, y/f - log(y/f) - 1. Tables
  give each model's mean QLIKE **relative to mcsGARCH, in percent**: -3.9% means 3.9% lower (better) than
  mcsGARCH.
* **p-values** come from a one-sided Diebold-Mariano test on daily average losses (Newey-West variance,
  Harvey-Leybourne-Newbold correction, Student t), exactly as in the Nifty study: a small p means the model's
  lower loss is unlikely to be luck. "<0.001" means below 0.001; numbers like 5e-12 mean 5 times 10 to the
  power -12.
* **Holm adjustment** raises p-values when several tests are made together, so that the chance of even one
  false alarm stays at 5%.
* **Walk-forward.** The first 150 days (2 January to 9 August 2023) are only for training. Then every model,
  mcsGARCH included, is refitted on all earlier days before each block of 21 trading days and forecasts that
  block. Test days: 660, from 10 August 2023 to 22 April 2026 (the last day of the Bank Nifty bars).
* **Model names** are those of `docs/MODEL_ITERATIONS_REAL.md`. The registered model is `hybrid_base` ("hybrid,
  new base"): mcsGARCH's forecast times a LightGBM correction, no GEX.

## 3. The primary test, H1 (REAL DATA)

| Horizon | Test days | mcsGARCH mean QLIKE | hybrid mean QLIKE | Change | DM statistic | raw p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| next 15 minutes | 660 | 0.2482 | 0.2385 | -3.9% | 7.03 | 2.6e-12 | 5.3e-12 |
| next 30 minutes | 660 | 0.1968 | 0.1869 | -5.0% | 6.72 | 2.0e-11 | 2.0e-11 |

Both Holm-adjusted p-values are at most 0.05, so by the registered rule (prereg Section 5) the result is
**replicated**. Source: rows `all`, `qlike`, `mcsgarch vs hybrid_base` of `tests_next15.csv` and
`tests_next30.csv`; Holm in `p91/hypotheses.csv`.

The seed check agrees: rerun with three single LightGBM seeds at 30 minutes, the hybrid is 4.9% to 5.1% better
than mcsGARCH (the registered 5-seed average: 5.0%); the largest gap between two seeds is 0.18 points
(`seed_noise_next30.csv`). Seed noise is far smaller than the gain.

## 4. Secondary hypotheses (REAL DATA, each reported whatever its direction)

### 4.1 H2, the last hour: success

| Window | Test days | mcsGARCH mean QLIKE | hybrid mean QLIKE | Change | DM statistic | p |
| --- | --- | --- | --- | --- | --- | --- |
| 14:30 to 15:30 | 660 | 0.1585 | 0.1456 | -8.1% | 5.03 | 3.2e-7 |

### 4.2 H3, Bank Nifty expiry days: success

| Horizon | Expiry days | mcsGARCH | hybrid | Change | DM statistic | raw p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| next 15 minutes | 84 | 0.2612 | 0.2539 | -2.8% | 2.60 | 0.0055 | 0.011 |
| next 30 minutes | 84 | 0.2022 | 0.1953 | -3.4% | 2.32 | 0.0115 | 0.011 |
| last hour (no hypothesis) | 84 | 0.1092 | 0.1124 | **+3.0%** (worse) | -0.47 | 0.68 | |

Both Holm p are at most 0.05, so H3 succeeds. The last hour of expiry days is the one place where the hybrid
was not better: on those 84 single windows mcsGARCH had the lower loss, though not significantly (the p for
"mcsGARCH is better" is 0.32). On Nifty the hybrid was 7.9% better there.

### 4.3 H4, the GEX ablation: prediction failed

The test: the same hybrid trained once without and once with each set of gamma-exposure (GEX) features. Negative
change = adding GEX lowered the loss. The prediction was that none of the 12 Holm-adjusted p-values (4 sets x 3
targets, Holm over the 4 sets within each target) would be at most 0.05. Four were, all at 15 minutes.

| Target, all 660 days | gross (no sign) | US convention | participant (Pro) | client mirror | placebo (wrong days) |
| --- | --- | --- | --- | --- | --- |
| next 15 minutes | **-0.84%** (p 0.002, Holm **0.009**) | **-0.67%** (0.012, Holm **0.024**) | **-0.56%** (0.027, Holm **0.027**) | **-0.63%** (0.007, Holm **0.021**) | +0.12% (0.82) |
| next 30 minutes | -0.21% (0.23, Holm 0.68) | -0.38% (0.11, Holm 0.46) | +0.29% (0.83, Holm 0.83) | -0.11% (0.35, Holm 0.70) | +0.20% (0.84) |
| last hour | +0.02% (0.51, Holm 1.00) | -0.20% (0.39, Holm 1.00) | +0.37% (0.69, Holm 1.00) | -0.70% (0.18, Holm 0.71) | +0.16% (0.62) |

**Next to the placebo**, as registered. The placebo set has the same columns as the US set but takes them from
other days, so it costs the model the same extra columns without real information. At 15 minutes the placebo
did not help (+0.12%, p 0.82), and the real US set beat the placebo by 0.79% (p 0.003). So the 15-minute gain
is not just the cost or luck of extra columns. At 30 minutes the real set also beat the placebo (0.57%, p 0.043)
although neither beat the base model. The verdict stays **"prediction failed"**.

On expiry days (reported without a verdict): at 15 minutes the gross set was 0.88% better (Holm 0.038), the
other three 0.40% to 0.60% (Holm 0.18); at 30 minutes and in the last hour no set had a Holm p below 0.29.

**Where the GEX gain comes from (not planned; exploratory).** The pipeline also splits the ablation by anchor
time and period (`ablation_next15.csv`):

* At the **09:15 forecast** (the first 15 minutes of the day, which absorb the overnight news) all four sets were
  7.5% to 8.6% better than the base hybrid (Holm within the anchor group 0.004 to 0.006); the placebo +0.03%.
  At **every later anchor** the four sets changed the loss by -0.15% to +0.13% (none significant). At 30 minutes,
  09:15 only: -1.9% to -4.2% (Holm 0.07 to 0.17). On Nifty, at 09:15 and 15 minutes, the same sets were 0.5% to
  2.2% better, none significantly.
* By period, the gain was larger later: from September 2025 the four sets were 1.3% to 2.3% better at 15
  minutes, before November 2024 0.1% to 0.4%.
* With a Holm adjustment over all 12 cells at once (stricter than registered), only the gross set at 15 minutes
  stays below 0.05 (0.027).

Because the gross set (no dealer sign at all) helps as much as the signed ones, this looks like information in
the *size* of option positioning, not in which way dealers lean. Since this was found after looking, it is a lead
for a new pre-registered test, not a finding.

## 5. Every model against mcsGARCH (REAL DATA)

Cells: mean QLIKE relative to mcsGARCH, and in brackets the one-sided DM p that the model beats mcsGARCH.
660 test days, 84 of them Bank Nifty expiry days. The bold row is the registered model.

| Model | 15 min, all days | 15 min, expiry days | 30 min, all days | 30 min, expiry days | last hour, all days | last hour, expiry days |
| --- | --- | --- | --- | --- | --- | --- |
| naive (last window) | +185.6% (1.000) | +188.4% (1.000) | +93.6% (1.000) | +99.7% (1.000) | +94.5% (1.000) | +451.2% (1.000) |
| naive (same window yesterday) | +226.4% (1.000) | +244.2% (1.000) | +198.0% (1.000) | +225.2% (1.000) | +159.6% (1.000) | +402.6% (0.999) |
| mcsGARCH refitted every 63 days | +0.9% (0.994) | +2.3% (0.976) | +1.7% (0.992) | +4.4% (0.969) | +4.5% (0.982) | +10.2% (0.925) |
| HAR | +6.6% (1.000) | +2.4% (0.841) | +5.0% (0.987) | +1.5% (0.684) | -7.8% (0.040) | +16.5% (0.875) |
| VIX | +71.4% (1.000) | +74.8% (1.000) | +82.6% (1.000) | +91.0% (1.000) | +94.5% (1.000) | +175.9% (1.000) |
| implied (straddle) | +48.4% (1.000) | +55.6% (1.000) | +52.8% (1.000) | +65.8% (1.000) | +71.7% (1.000) | +155.9% (1.000) |
| mcsGARCH, implied daily level | -2.8% (0.005) | -0.2% (0.456) | -4.0% (0.011) | +1.1% (0.663) | -4.1% (0.173) | +5.5% (0.655) |
| ridge log-HAR, v1 base | +2.0% (0.918) | -0.5% (0.390) | -0.5% (0.382) | -0.9% (0.349) | -5.0% (0.022) | +5.1% (0.758) |
| ridge log-HAR, new base | -0.4% (0.382) | -0.6% (0.367) | -3.1% (0.026) | -0.5% (0.418) | -11.4% (<0.001) | -6.1% (0.242) |
| LightGBM, v1 base | -2.1% (0.217) | -3.4% (0.043) | -3.4% (0.194) | -3.9% (0.061) | -7.2% (0.025) | +20.0% (0.964) |
| LightGBM, new base | -3.7% (0.024) | -4.1% (0.003) | -4.8% (0.063) | -4.8% (0.014) | -7.1% (0.038) | +17.2% (0.917) |
| LightGBM tuned, new base | -3.8% (0.002) | -4.3% (0.003) | -3.9% (0.106) | -2.9% (0.102) | -7.0% (0.086) | +18.1% (0.908) |
| hybrid, v1 base | -4.3% (<0.001) | -3.2% (0.001) | -5.5% (<0.001) | -3.9% (0.002) | -8.7% (<0.001) | +2.7% (0.659) |
| hybrid, v1 base + v1 GEX | -4.5% (<0.001) | -2.7% (0.009) | -5.0% (<0.001) | -3.1% (0.017) | -7.5% (<0.001) | +4.6% (0.743) |
| **hybrid, new base** | -3.9% (<0.001) | -2.8% (0.006) | -5.0% (<0.001) | -3.4% (0.011) | -8.1% (<0.001) | +3.0% (0.678) |
| hybridiv, new base | -5.7% (<0.001) | -3.6% (0.023) | -7.5% (<0.001) | -3.3% (0.119) | -11.6% (0.001) | +12.2% (0.813) |

The models with GEX (each base model plus one GEX set):

| Model | 15 min, all days | 15 min, expiry days | 30 min, all days | 30 min, expiry days | last hour, all days | last hour, expiry days |
| --- | --- | --- | --- | --- | --- | --- |
| `linear_gex_gross` | -0.8% (0.266) | -0.6% (0.364) | -3.2% (0.026) | -0.6% (0.404) | -12.1% (<0.001) | -2.6% (0.384) |
| `linear_gex_us` | -1.2% (0.191) | -1.0% (0.290) | -3.8% (0.012) | -1.4% (0.297) | -12.1% (<0.001) | +1.0% (0.543) |
| `linear_gex_participant` | -0.6% (0.310) | +0.0% (0.510) | -2.9% (0.044) | +0.7% (0.602) | -12.1% (<0.001) | -0.1% (0.495) |
| `linear_gex_client` | -0.7% (0.282) | -0.4% (0.414) | -3.0% (0.031) | +0.8% (0.610) | -11.0% (0.001) | +0.1% (0.505) |
| `linear_gex_placebo` | -0.2% (0.440) | +0.1% (0.535) | -2.3% (0.078) | +1.4% (0.714) | -10.0% (0.002) | -3.2% (0.368) |
| `hybrid_gex_gross` | -4.7% (<0.001) | -3.7% (<0.001) | -5.2% (<0.001) | -4.3% (0.003) | -8.1% (<0.001) | +6.1% (0.804) |
| `hybrid_gex_us` | -4.6% (<0.001) | -3.2% (0.003) | -5.4% (<0.001) | -4.0% (0.003) | -8.3% (<0.001) | +3.6% (0.719) |
| `hybrid_gex_participant` | -4.5% (<0.001) | -3.4% (0.002) | -4.8% (<0.001) | -3.1% (0.015) | -7.8% (<0.001) | +6.3% (0.808) |
| `hybrid_gex_client` | -4.5% (<0.001) | -3.4% (0.001) | -5.1% (<0.001) | -3.6% (0.009) | -8.8% (<0.001) | +1.8% (0.614) |
| `hybrid_gex_placebo` | -3.8% (<0.001) | -2.3% (0.016) | -4.8% (<0.001) | -3.5% (0.006) | -8.0% (<0.001) | +6.1% (0.819) |
| `hybridiv_gex_gross` | -6.5% (<0.001) | -4.1% (0.011) | -7.9% (<0.001) | -3.4% (0.125) | -11.9% (<0.001) | +12.1% (0.807) |
| `hybridiv_gex_us` | -6.5% (<0.001) | -4.0% (0.018) | -7.8% (<0.001) | -2.8% (0.165) | -11.6% (0.001) | +10.3% (0.773) |
| `hybridiv_gex_participant` | -6.4% (<0.001) | -4.0% (0.015) | -8.1% (<0.001) | -3.4% (0.111) | -11.2% (0.002) | +13.1% (0.816) |
| `hybridiv_gex_client` | -6.5% (<0.001) | -4.2% (0.010) | -7.9% (<0.001) | -3.2% (0.125) | -11.9% (<0.001) | +9.3% (0.761) |
| `hybridiv_gex_placebo` | -5.6% (<0.001) | -3.5% (0.023) | -7.0% (<0.001) | -3.0% (0.137) | -11.0% (0.001) | +13.1% (0.835) |

What stands out (descriptive, no hypothesis):

* The picture is the same as on Nifty: simple tools (HAR, VIX, the straddle alone, the 63-day refit) mostly lose
  to mcsGARCH, and the hybrids win.
* **The registered hybrid is not the best model on Bank Nifty.** `hybridiv` (the same correction on top of an
  mcsGARCH whose daily level comes from option prices) is better at every all-days horizon: 5.7%, 7.5% and 11.6%
  against 3.9%, 5.0% and 8.1%. On Nifty it was about equal at 15 and 30 minutes and worse in the last hour, which
  is why the Nifty study chose `hybrid_base`. This was not a hypothesis and does not change the verdict.
* **Last hour on expiry days:** no model with a learned correction beats mcsGARCH significantly. With only 84
  single windows these cells are noisy.

## 6. Breakdowns (REAL DATA, no hypotheses)

### 6.1 By expiry regime, and weekly against monthly-only options

The pipeline's three regimes (cut at 20 November 2024, when Bank Nifty's weekly options were gone, and
1 September 2025, when its monthly expiry moved to Tuesday). Hybrid against mcsGARCH:

| Period | Test days | 15 min | 30 min | last hour |
| --- | --- | --- | --- | --- |
| weekly options (to 19 Nov 2024) | 312 | -3.3% (<0.001) | -3.4% (<0.001) | -3.8% (0.042) |
| monthly only, 20 Nov 2024 to 31 Aug 2025 | 192 | -5.3% (<0.001) | -7.6% (<0.001) | -10.5% (<0.001) |
| monthly only, from 1 Sep 2025 | 156 | -3.4% (0.018) | -5.5% (0.004) | -14.6% (<0.001) |
| **monthly only, pooled** (from 20 Nov 2024) | 348 | -4.5% (6e-8) | -6.8% (2e-10) | -12.2% (3e-8) |

On expiry days only:

| Period | Expiry days | 15 min | 30 min | last hour |
| --- | --- | --- | --- | --- |
| weekly options (to 19 Nov 2024) | 67 | -3.0% (0.012) | -3.6% (0.021) | +9.1% (0.90) |
| monthly only (from 20 Nov 2024) | 17 | -2.3% (0.098) | -2.6% (0.16) | -18.1% (0.051) |

The hybrid's gain is larger in the monthly-only period than in the weekly period, and on non-expiry days
(15 minutes: 3.4% weekly, 5.5% and 3.4% in the two later regimes; 30 minutes: 3.3%, 8.0% and 5.7%), not only on
the 17 monthly expiry days, which are too few to test. (Sources: `scoreboard_<target>.csv` and
`tests_<target>.csv` for the regimes; `p91/period_split.csv` for the pooled split, computed with the pipeline's
`daily_losses` and `diebold_mariano`.)

### 6.2 The 09:15 forecast against later ones

| Model | 15 min, 09:15 only | 15 min, without 09:15 | 30 min, 09:15 only | 30 min, without 09:15 |
| --- | --- | --- | --- | --- |
| HAR | +33.5% (1.000) | +4.1% (0.995) | +28.8% (1.000) | +2.8% (0.887) |
| LightGBM, new base | -8.8% (0.077) | -3.2% (0.054) | -13.5% (0.009) | -3.9% (0.122) |
| **hybrid, new base** | -7.6% (0.069) | -3.6% (<0.001) | -11.7% (0.014) | -4.4% (<0.001) |
| hybridiv, new base | -11.1% (0.033) | -5.2% (<0.001) | -12.3% (0.050) | -7.0% (<0.001) |

As on Nifty, the gain is largest at the open, but most of the evidence comes from the later anchors: without
09:15 the hybrid is still 3.6% and 4.4% better with p below 0.001. (`gain_by_anchor_<target>.csv` has every
anchor minute.)

### 6.3 Trimmed QLIKE, MSE and the Model Confidence Set

| Model | trimmed QLIKE 15 min | trimmed 30 min | trimmed last hour | MSE 15 min (p) | MSE 30 min (p) | MSE last hour (p) |
| --- | --- | --- | --- | --- | --- | --- |
| HAR | +5.4% | +3.7% | -9.5% | -35.7% (0.168) | -32.9% (0.176) | -6.8% (0.271) |
| mcsGARCH, implied daily level | -2.4% | -3.6% | -2.2% | -43.5% (0.162) | -46.8% (0.164) | +4.1% (0.717) |
| ridge log-HAR, new base | -1.6% | -4.6% | -10.5% | -41.0% (0.157) | -44.7% (0.151) | -33.4% (0.046) |
| LightGBM, new base | -5.5% | -8.0% | -10.7% | -35.5% (0.153) | -31.4% (0.163) | +19.4% (0.769) |
| **hybrid, new base** | -4.8% | -6.3% | -9.5% | -9.8% (0.118) | +1.0% (0.631) | -8.8% (0.074) |
| hybridiv, new base | -6.5% | -8.8% | -11.9% | -45.1% (0.153) | -49.3% (0.152) | -18.9% (0.046) |

* **Trimmed QLIKE** (each model's own largest 1% of losses removed): the hybrid stays 4.8%, 6.3% and 9.5%
  better, so the win is not driven by a few extreme windows.
* **MSE** (squared error, dominated by a few wild windows): the registered hybrid does not beat mcsGARCH
  significantly at any horizon (p 0.118, 0.631 and 0.074). Of all 31 models, only two GEX hybrids do so at
  15 minutes (`hybrid_gex_gross` 3.1% better, p 0.008; `hybrid_gex_participant` 4.8%, p 0.046), none at 30
  minutes, and 17 in the last hour, mostly with p between 0.01 and 0.05. As on Nifty, QLIKE and MSE tell
  different stories, and MSE is very noisy.
* **Model Confidence Set** (models that cannot be separated from the best at the 10% level, `mcs_*.csv`): under
  QLIKE on all days, `hybrid_base` is in the set at all three horizons and mcsGARCH is out (sets of 16, 15 and 24
  of 32 models). On expiry days `hybrid_base` is in at all three horizons; mcsGARCH is out at 15 minutes and in
  at 30 minutes and the last hour. Under MSE both are in everywhere.

### 6.4 "Development" and "hold-out" periods

For Bank Nifty all test days are out of sample for the design, so this split (at 1 July 2025) has no special
meaning; it is reported because the pipeline prints it.

| Model | 15 min, to 30 Jun 2025 (462 days) | 15 min, from 1 Jul 2025 (198) | 30 min, to 30 Jun 2025 | 30 min, from 1 Jul 2025 | last hour, to 30 Jun 2025 | last hour, from 1 Jul 2025 |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM, new base | -3.3% (0.089) | -5.0% (0.001) | -3.3% (0.204) | -9.8% (<0.001) | -1.7% (0.368) | -20.9% (<0.001) |
| LightGBM tuned, new base | -3.6% (0.015) | -4.3% (0.002) | -2.4% (0.273) | -8.9% (<0.001) | -0.5% (0.468) | -23.4% (<0.001) |
| **hybrid, new base** | -3.8% (<0.001) | -4.2% (<0.001) | -4.3% (<0.001) | -7.6% (<0.001) | -4.6% (0.003) | -17.0% (<0.001) |
| hybridiv, new base | -5.6% (<0.001) | -6.0% (<0.001) | -6.6% (<0.001) | -10.4% (<0.001) | -7.0% (0.066) | -23.2% (<0.001) |

### 6.5 Seed noise (30 minutes)

Three single seeds, against mcsGARCH: `hybrid_base` -5.05%, -4.93%, -5.10%; `lgbm_base` -4.98%, -5.02%, -4.72%;
`hybrid_gex_participant` -4.50%, -4.94%, -4.87%; `hybridiv_base` -7.41%, -7.13%, -7.51%. The largest gap
between two seeds of one model is 0.45 points (`seed_noise_next30.csv`). The registered forecasts average 5
seeds, which shrinks this further.

## 7. Nifty on the same dates (REAL DATA)

The Nifty v2 forecasts (`results/pipeline_nse_v2/forecasts_<target>.csv`, not rerun) restricted to the same 660
test dates and scored the same way. Each index uses its own expiry days. The two models were trained on each
index's own walk-forward, so this is a description, not a test of the difference.

| Target | Nifty, all days | Bank Nifty, all days | Nifty, its 142 expiry days | Bank Nifty, its 84 expiry days |
| --- | --- | --- | --- | --- |
| next 15 minutes | -5.5% (DM 9.05) | -3.9% (DM 7.03) | -5.2% (DM 3.44) | -2.8% (DM 2.60) |
| next 30 minutes | -6.9% (DM 7.76) | -5.0% (DM 6.72) | -5.4% (DM 2.89) | -3.4% (DM 2.32) |
| last hour | -5.7% (DM 2.43) | -8.1% (DM 5.03) | -7.0% (DM 1.93) | +3.0% (DM -0.47) |

At 15 and 30 minutes the Bank Nifty gain is about 70% of Nifty's; in the last hour it is larger. (On these
dates Nifty's numbers are a little smaller than its full-sample 5.8%, 7.6% and 6.4%, which include its later
days and the closing-auction period.) Source: `p91/nifty_same_dates.csv`.

## 8. Data checks (recomputed by the run; all match the pre-registration)

* 810 trading days, 2 January 2023 to 22 April 2026 (`daily_market.csv`, `run_info.json`). Training to
  9 August 2023; 660 test days from 10 August 2023; 25, 24 and 1 forecasts a day for the three targets.
* No day without a Bank Nifty bhavcopy, without India VIX or without Bank Nifty futures rows.
* Test days by regime: 312, 192 and 156. Bank Nifty expiry days among them: 84 (32 monthly), 67 in the weekly
  period and 17 after it. Development 462 days, hold-out 198.
* Daily GEX panel: the only missing values are the Pro-rule gamma-flip level (523 of 810 days), as expected.
* The stdout log (`results/pipeline_nse_banknifty_stdout.txt`) shows no warning or skipped step. The run took
  78 minutes (4,658 seconds).
* **One fact worth knowing about the benchmark.** mcsGARCH's daily level comes from an EGARCH model on daily
  returns. When that fit fails, the code (unchanged from Nifty) falls back to a simple EWMA variance. On Bank
  Nifty this happened in 8 of the 32 refits (the first six, 10 August 2023 to 15 January 2024, and those of
  22 April and 23 May 2024), against 2 of 38 on Nifty (`mcsgarch_params.csv`, column `daily_method`). This did
  not flatter the hybrid: in the 168 test days forecast with the EWMA fallback the hybrid's gain was smaller
  (3.0%, 4.2% and 3.5%) than in the other 492 days (4.3%, 5.3% and 10.3%). (Exploratory split, computed with the
  same functions.)

## 9. Fairness caveats: what this result does and does not show

1. **Bank Nifty is not independent of Nifty.** The test uses the same 660 dates, the same news, the same India
   VIX and the same participant file, and the two indices' returns are highly correlated (their last-30-minute
   returns correlate 0.833, `docs/FALSIFICATION.md` T5). A success shows that the design is not specific to
   Nifty's data. It is not a second, independent sample in time.
2. **India VIX stands in for a Bank Nifty VIX.** NSE publishes no Bank Nifty volatility index, so the `log_vix`
   feature and the `vix` model use India VIX, which is built from Nifty options. This should, if anything, have
   made the hybrid weaker on Bank Nifty. Bank Nifty's own option prices enter through the implied block.
3. **The option features mean something different after November 2024.** Bank Nifty's last weekly options
   expired on 13 November 2024. From then on the "nearest expiry" contract that feeds GEX and the near-expiry
   implied features is a monthly up to 25 trading days away, so those features describe the rest of the month,
   not the next day. Nothing was corrected for this. The hybrid's gain was in fact larger in the monthly-only
   period (Section 6.1).
4. **mcsGARCH's fixed expiry curve on monthly expiries.** mcsGARCH learns one intraday shape for expiry days
   from all expiry days in its training window. After November 2024 that window is mostly weekly expiries, but it
   is applied to monthly ones. The hybrid's calendar features can repair that shape, which would be an easy win
   rather than new information. The data say this is not where the main result comes from: on the 17
   monthly-only expiry days the gains at 15 and 30 minutes (2.3% and 2.6%) were smaller than on all days. The
   last hour on those 17 days (18.1% better, p 0.051) may partly be this repair; it is too few days to tell.
5. **The project was not blind to Bank Nifty.** Earlier studies computed Bank Nifty volatility and GEX results
   (listed in the pre-registration), although none made a forecast and none shaped the hybrid.
6. **A shorter sample with no closing-auction days,** and third-party bars of slightly lower measured quality
   than Nifty's (high and low within 5 basis points of NSE's on 99.4% and 99.3% of days, against 100% and 99.9%).
7. **The benchmark's daily level fell back to EWMA in 8 of 32 refits** (Section 8). This did not inflate the
   hybrid's win, but it means the benchmark was not always the paper's exact model.

## 10. Deviations from the pre-registration

**None in the registered analysis.** The command of prereg Section 1 was run unchanged, nothing crashed, and no
model, feature, setting, sample, test or threshold was changed. For completeness:

* **How the code was loaded.** The package is installed in editable mode from the main checkout, which was busy
  with another run. The command was therefore run from a separate worktree at commit 97eb0a1 (the pre-registration
  commit) with `PYTHONPATH` set to that worktree's `src`, so that exactly the committed code was used. This changes
  no number.
* **One helper script was added**, `experiments/p91_banknifty_summary.py`. It fits nothing: it reads the pipeline's
  outputs and computes what the pipeline does not write itself but the plan asks for (Holm over the two H1 tests,
  the verdicts, the pooled weekly against monthly-only split, and Nifty on the same dates), with the pipeline's own
  functions. For "Nifty on the same dates" each index's expiry rows use its own expiry days; the plan did not say
  which expiry days to use there.
* **Labelled exploratory looks** that were not in the plan, reported because they help to read the result: the
  GEX ablation by anchor time and period with Holm over all 12 cells (Section 4.3), the regime split into expiry
  and other days (Section 6.1), and the EWMA-fallback split (Section 8). None of them changes a verdict.

## 11. Files and how to reproduce

* Plan: `docs/prereg/P91_BANKNIFTY_FORECAST.md`.
* Pipeline outputs (REAL DATA): `results/pipeline_nse_banknifty/` (forecasts, scoreboards, DM tests, ablation
  tables, Model Confidence Sets, mcsGARCH parameters, seed noise, figures, `run_info.json`); log
  `results/pipeline_nse_banknifty_stdout.txt`.
* Registered summaries: `results/pipeline_nse_banknifty/p91/` (`hypotheses.csv`, `h4_ablation_hybrid.csv`,
  `scoreboard_vs_mcsgarch.csv`, `period_split.csv`, `nifty_same_dates.csv`, `mcs_membership.csv`,
  `data_checks.json`).
* Reproduce, from the repository root:

```
OMP_NUM_THREADS=1 python scripts/run_pipeline.py --source nse --symbol BANKNIFTY --end 2026-04-22 \
    --n-seeds 5 --seed-study 3 --verbose --out results/pipeline_nse_banknifty \
    > results/pipeline_nse_banknifty_stdout.txt 2>&1
python experiments/p91_banknifty_summary.py
```
