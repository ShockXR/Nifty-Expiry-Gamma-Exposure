# Live dry run: Nifty expiry on 2026-11-03 (scored)

**REAL DATA, live demonstration (n = 1, not evidence).** Forecasts were frozen in `forecast.json` before the open.

| Item | Forecast | Outcome | Note |
| --- | --- | --- | --- |
| Whole-day variance | 5.551e-05 (80%: 2.69e-05 to 8.60e-05) | 3.595e-05 | inside the interval; QLIKE 0.082 |
| Last-hour variance | 1.334e-05 (80%: 4.68e-06 to 2.22e-05) | 1.357e-05 | inside the interval; QLIKE 0.000 |
| Last hour, 14:30 rule | 9.973e-06 (80%: 4.21e-06 to 1.65e-05) | 1.357e-05 | inside; QLIKE 0.053 |
| Distance of the official close from the straddle strike 23,350 | 152 points (straddle price) | 461 points | |
| Last 30 minutes (R5 rule) | 1.0 bp | 48.5 bp | rest of day 108.2 bp; same sign: True |
| Distance of close to nearest 50-point strike | about 12.5 points on average | 10.85 points (23,800) | within 5 points: False |

One day cannot confirm or refute any of these models. The value of this exercise is that the whole chain (download, features, frozen models, scoring) ran on live data with the forecasts committed before the outcome existed.
