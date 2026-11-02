# Live dry run: Nifty expiry on 2026-11-03 (frozen forecasts)

**REAL DATA, live demonstration (n = 1, not evidence).** Information set: files published up to the close of 2026-11-02. Frozen at 2026-11-02 21:01:17 UTC; the git commit that adds this file is the timestamp that matters.

## 1. Volatility, day ahead (HAR-style regressions on all past days)

| Target | Forecast (variance) | As annualised vol | 80% interval (annualised vol) | Training days |
| --- | --- | --- | --- | --- |
| Whole day, 09:15 to 15:29 | 5.551e-05 | 11.8% | 8.2% to 14.7% | 920 |
| Last hour, 14:30 to 15:29 | 1.334e-05 | 14.5% | 8.6% to 18.7% | 920 |

(The last-hour row is scaled to a full-day rate so the two are comparable.)

## 2. What the option market expects

Forward from put-call parity 23,329.4; at-the-money strike 23,350; the expiring straddle costs 152.1 points, so the market expects the official close to land on average about **152 points** away from 23,350. The previous official close was 23,440.80: since 3 August 2026 the index close comes from a closing auction, while option settlement prices come from F&O trading, which is the likely reason the forward and the official close differ. Implied variance to the close 6.677e-05 (includes the overnight gap, so it is not directly comparable with section 1).

## 3. Dealer gamma at the previous close

| Measure | Value |
| --- | --- |
| Gross GEX (crore per 1% move) | 128,342 |
| Net GEX, US convention | 2,816 |
| Net GEX, Pro participant | 2,382 |
| Net GEX, Client mirror | 5,926 |
| Gamma flip (US convention) | 23,320 |
| Call wall / put wall | 24,000 / 23,000 |
| Max open-interest strike / max pain | 23,000 / 23,250 |

* **R3 (in-sample, expiry days):** Higher GEX the evening before went with a MORE volatile expiry day under the Pro and client-mirror rules; all three rules are positive tonight.
* **R5 (in-sample):** End-of-day reversal was concentrated after negative Pro GEX; tonight Pro GEX is positive, so little follow-through either way is expected.
* **Pinning study:** No dealer-gamma proxy explained pinning; in the Tuesday regime there was no pinning on average.

## 4. Frozen rules applied during the day

* **14:30 rule (last hour):** log last-hour variance = frozen coefficients x (log variance from 09:15 to 14:30 today, yesterday's last hour, yesterday's VIX, regime and expiry terms); 941 training days, R squared 0.42.
* **End-of-day rule (R5), state Pro GEX >= 0:** predicted last-30-minute return = -0.16 bp + 0.0107 x (return from the previous close to 14:59, in bp); 929 training days.

## 5. Pinning

In the Tuesday regime (61 expiries) the official close was on average 14.6 points from the nearest 50-point strike on expiry days and 13.4 on other days (12.5 means no clustering). Forecast: No pinning: distance to the nearest 50-point strike drawn from the same distribution as other days (mean about 12.5 points, 20% chance of being within 5 points).
