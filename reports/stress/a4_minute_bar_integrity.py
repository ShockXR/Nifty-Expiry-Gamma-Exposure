"""A4. Integrity of the REAL Nifty 50 1-minute bars fetched with scripts/fetch_minute_bars.py (scratch copy), versus the committed data-quality reports.
Usage: python -I a4_minute_bar_integrity.py <raw_dir> <repo> <out.csv>"""
import sys, numpy as np, pandas as pd
raw, repo, out = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(raw + "/nifty_1min.csv"); rows = []
def add(check, value, committed="", status=""): rows.append(dict(check=check, value=value, committed=committed, status=status)); print(check, "|", value, "|", committed, status)
t = pd.to_datetime(df.timestamp); df["t"] = t; df["day"] = t.dt.date
add("rows", len(df), "886,923 (minute_bars_source.json)", "VERIFIED" if len(df) == 886923 else "FAILED")
add("duplicate timestamps", int(t.duplicated().sum()))
add("timestamps strictly increasing in file order", bool(t.is_monotonic_increasing and not t.duplicated().any()))
add("weekend bars", int((t.dt.day_of_week >= 5).sum()))
mod = t.dt.hour * 60 + t.dt.minute
add("bars outside 09:15 to 15:29", int(((mod < 555) | (mod > 929)).sum()))
g = df.groupby("day").size(); add("trading days / median bars per day / days with 375 bars", f"{len(g)} / {int(g.median())} / {int((g == 375).sum())}", "dq1 coverage: 2368 days")
add("days with fewer than 370 bars", int((g < 370).sum()), "1 since 2023 (data_validation summary)"); add("days with more than 375 bars", int((g > 375).sum()))
short = g[g < 375]; add("days with 1 to 369 bars (dates)", ",".join(f"{d}:{n}" for d, n in short.items() if n < 370)[:200])
gap = df.groupby("day").t.diff().dt.total_seconds().div(60); add("within-session gaps (diff > 1 minute) count / days affected", f"{int((gap > 1).sum())} / {df.day[gap > 1].nunique()}")
add("NaN or non-positive price", int((df[["open", "high", "low", "close"]].isna() | (df[["open", "high", "low", "close"]] <= 0)).sum().sum()))
bad = (df.high < df.low) | (df.close > df.high) | (df.close < df.low) | (df.open > df.high) | (df.open < df.low); add("OHLC inconsistent bars", int(bad.sum()))
df["flat"] = df.high == df.low; df["zero_ret"] = df.close.diff().fillna(1) == 0
fs = df.groupby("day").flat.mean(); zs = df.groupby("day").zero_ret.mean()
cov = pd.read_csv(repo + "/results/dq1_flat_bars/coverage.csv").set_index("index").loc["nifty"]
add("days with more than 5% / 20% bars with high == low (stricter definition, not the committed one)", f"{int((fs > 0.05).sum())} / {int((fs > 0.2).sum())}")
add("median share of zero close-to-close returns per day", float(zs.median()), f"dq1 coverage median_flat_share: {cov.median_flat_share}")
add("days with more than 5% / 20% zero close-to-close returns", f"{int((zs > 0.05).sum())} / {int((zs > 0.2).sum())}", f"dq1 coverage: {int(cov.days_over_5pct)} / {int(cov.days_over_20pct)}", "VERIFIED" if (int((zs > 0.05).sum()), int((zs > 0.2).sum())) == (int(cov.days_over_5pct), int(cov.days_over_20pct)) else "DIFFERS")
lr = np.log(df.close).groupby(df.day).diff().dropna(); sd = np.median(np.abs(lr)) * 1.4826
add("returns beyond 20 robust sd (count) / largest |r| in sd", f"{int((np.abs(lr) > 20 * sd).sum())} / {float(np.abs(lr).max() / sd):.1f}")
d23 = df[df.t >= "2023-01-02"]; add("rows since 2023-01-02 / days", f"{len(d23)} / {d23.day.nunique()}", "943 compared days in data_validation summary")
pd.DataFrame(rows).to_csv(out, index=False)
