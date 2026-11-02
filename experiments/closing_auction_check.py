"""Robustness check: NSE's closing auction (from 3 August 2026). REAL DATA.

From 3 August 2026 NSE sets the official close of Nifty 50 in a closing
auction. In the 1-minute bars this shows up as an index that stands still
from 15:15 to 15:28 and then jumps to the auction price in the 15:29 bar
(that bar's close equals NSE's official close on every one of these days).
Before that date the last bar's close is the last trade and the minutes
after 15:15 trade normally.

That change alters what "the last hour" and "the last 30 minutes" measure
for the last 63 trading days of the sample, so this script reruns three
results without those days:

* R1: the forecast scoreboards (QLIKE vs mcsGARCH) on test days before
  3 August 2026 only, from results/pipeline_nse/forecasts_*.csv;
* R2: the week-relative expiry-day effect in the Tuesday regime, split
  into before and after the auction start;
* R5: the end-of-day momentum regressions without the auction days.

It also writes the evidence for the auction itself (frozen minutes, size of
the 15:29 jump, match of the 15:29 close with NSE's official close).

    python experiments/closing_auction_check.py

Writes results/closing_auction/*.csv and results/closing_auction_stdout.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import momentum_gex as mg  # noqa: E402
import real_day_level as rdl  # noqa: E402

from niftygex.evaluation import daily_losses, diebold_mariano, qlike  # noqa: E402

AUCTION_START = pd.Timestamp("2026-08-03")
RUN, RAW, OUT = Path("results/pipeline_nse"), Path("data/raw"), Path("results/closing_auction")
MODELS = ["har", "vix", "lgbm_base", "lgbm_gex", "hybrid_base", "hybrid_gex", "hybrid_gex_us", "hybrid_gex_participant"]


def auction_evidence(dm: pd.DataFrame) -> pd.DataFrame:
    b = pd.read_csv(RAW / "nifty_1min.csv", parse_dates=["timestamp"])
    b = b[b["timestamp"] >= "2026-01-01"]
    b["day"] = b["timestamp"].dt.normalize()
    b["hm"] = b["timestamp"].dt.strftime("%H:%M")
    late = b[(b["hm"] >= "15:16") & (b["hm"] <= "15:28")]
    frozen = late.groupby("day").apply(lambda x: float((x["high"] == x["low"]).mean()), include_groups=False)
    c = b.pivot_table(index="day", columns="hm", values="close")
    jump = (np.log(c["15:29"] / c["15:28"]) * 1e4).abs()
    idx = pd.read_csv(RAW / "index_daily.csv", parse_dates=["date"])
    off = idx[idx["index"].str.upper() == "NIFTY 50"].set_index("date")["close"]
    d = pd.DataFrame({"frozen_share_1516_1528": frozen, "abs_jump_1529_bp": jump,
                      "last_bar_vs_official_bp": (dm["close_last_bar"] / off - 1).abs() * 1e4}).dropna()
    d["period"] = np.where(d.index >= AUCTION_START, "from 3 Aug 2026", "Jan to Jul 2026")
    return d.groupby("period").agg(days=("abs_jump_1529_bp", "size"),
                                   mean_frozen_share=("frozen_share_1516_1528", "mean"),
                                   mean_abs_jump_1529_bp=("abs_jump_1529_bp", "mean"),
                                   median_last_bar_vs_official_bp=("last_bar_vs_official_bp", "median")).reset_index()


def scoreboards() -> pd.DataFrame:
    rows = []
    for target in ("next15", "next30", "last_hour"):
        f = pd.read_csv(RUN / f"forecasts_{target}.csv", parse_dates=["date"])
        for sample, m in (("all test days", np.ones(len(f), bool)), ("before 3 Aug 2026", (f["date"] < AUCTION_START).to_numpy())):
            for group, g in (("all", m), ("expiry", m & f["is_expiry"].to_numpy(bool))):
                x = f[g]
                base = qlike(x["y"], x["mcsgarch"]).mean()
                for mod in MODELS:
                    if mod not in x:
                        continue
                    rows.append({"target": target, "sample": sample, "group": group, "model": mod, "anchors": len(x),
                                 "days": x["date"].nunique(), "qlike_vs_mcsgarch_pct": 100 * (qlike(x["y"], x[mod]).mean() / base - 1)})
    return pd.DataFrame(rows)


def dm_tests() -> pd.DataFrame:
    """One-sided Diebold-Mariano tests on daily mean QLIKE (is the model better than mcsGARCH?), as in run_pipeline."""
    rows = []
    for target in ("next15", "next30", "last_hour"):
        f = pd.read_csv(RUN / f"forecasts_{target}.csv", parse_dates=["date"])
        for sample, m in (("all test days", np.ones(len(f), bool)), ("before 3 Aug 2026", (f["date"] < AUCTION_START).to_numpy())):
            for group, g in (("all", m), ("expiry", m & f["is_expiry"].to_numpy(bool))):
                L = daily_losses(f[g], ["mcsgarch", "har", "lgbm_base", "hybrid_base"])
                for mod in ("har", "lgbm_base", "hybrid_base"):
                    rows.append({"target": target, "sample": sample, "group": group, "comparison": f"mcsgarch vs {mod}",
                                 **diebold_mariano(L["mcsgarch"], L[mod], alternative="greater")})
    return pd.DataFrame(rows)


def r2_split(dm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ("rv_day", "rv_last_hour"):
        wr = rdl.week_relative(dm, col)
        wr = wr[wr["regime"] == "single_weekly_tue"]
        for name, m in (("Tuesday regime, all", np.ones(len(wr), bool)),
                        ("Tuesday regime, before 3 Aug 2026", (wr["date"] < AUCTION_START).to_numpy()),
                        ("Tuesday regime, from 3 Aug 2026", (wr["date"] >= AUCTION_START).to_numpy())):
            x = wr.loc[m, "rel"].dropna().to_numpy()
            if len(x) < 3:
                continue
            boot = np.random.default_rng(0).choice(x, (4000, len(x))).mean(axis=1)
            from scipy.stats import wilcoxon
            rows.append({"measure": col, "sample": name, "weeks": len(x), "pct_expiry_vs_week": 100 * (np.exp(x.mean()) - 1),
                         "ci95_low_pct": 100 * (np.exp(np.percentile(boot, 2.5)) - 1),
                         "ci95_high_pct": 100 * (np.exp(np.percentile(boot, 97.5)) - 1),
                         "p_signed_rank": float(wilcoxon(x).pvalue)})
    return pd.DataFrame(rows)


def r5_without_auction(dm: pd.DataFrame) -> pd.DataFrame:
    panel = pd.read_csv(RUN / "daily_gex_panel.csv", index_col=0, parse_dates=True)
    d = dm.join(mg.daily_returns(RAW / "nifty_1min.csv", dm.index))
    scale = panel["gross_gex"].rolling(20, min_periods=5).median()
    for k, col in mg.RULES.items():
        d[f"gex_{k}"] = (panel[col] / scale).shift(1).reindex(d.index)
    d["lvix"] = np.log(d["vix"]).shift(1)  # shift on the full trading calendar, before dropping days
    d = d.dropna(subset=["r_rod", "r_last"])
    d["rr"], d["rl"] = d["r_rod"] * 1e4, d["r_last"] * 1e4
    d["lvix"] -= d["lvix"].mean()
    rows = []
    for sample, m in (("all days", np.ones(len(d), bool)), ("before 3 Aug 2026", (d.index < AUCTION_START))):
        dd = d[m]
        f = mg.fit(dd["rl"], dd[["rr"]])
        rows.append({"sample": sample, "spec": "momentum only", "rule": "none", "term": "r_rod", "coef": f.params["rr"],
                     "p": f.pvalues["rr"], "n": int(f.nobs)})
        for k in mg.RULES:
            neg = (dd[f"gex_{k}"] < 0).astype(float).where(dd[f"gex_{k}"].notna())
            X = pd.DataFrame({"r_rod": dd["rr"], "neg": neg, "r_rod_x_neg": dd["rr"] * neg})
            f = mg.fit(dd["rl"], X)
            rows.append({"sample": sample, "spec": "binary", "rule": k, "term": "r_rod_x_neg", "coef": f.params["r_rod_x_neg"],
                         "p": f.pvalues["r_rod_x_neg"], "n": int(f.nobs)})
            X = X.assign(lvix=dd["lvix"], r_rod_x_lvix=dd["rr"] * dd["lvix"], r_rod_x_abs=dd["rr"] * dd["rr"].abs())
            f = mg.fit(dd["rl"], X)
            rows.append({"sample": sample, "spec": "binary + VIX and size controls", "rule": k, "term": "r_rod_x_neg",
                         "coef": f.params["r_rod_x_neg"], "p": f.pvalues["r_rod_x_neg"], "n": int(f.nobs)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dm = pd.read_csv(RUN / "daily_market.csv", index_col=0, parse_dates=True)
    parts = {"auction_evidence": auction_evidence(dm), "r1_scoreboards": scoreboards(), "r1_dm_tests": dm_tests(),
             "r2_tuesday_split": r2_split(dm), "r5_without_auction": r5_without_auction(dm)}
    lines = ["REAL DATA. Robustness to NSE's closing auction (official close set by auction from 3 Aug 2026).", ""]
    for name, df in parts.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        show = df
        if name == "r1_scoreboards":
            show = df[df["model"].isin(["har", "lgbm_base", "hybrid_base", "hybrid_gex_participant"])]
        with pd.option_context("display.width", 200, "display.max_rows", 200, "display.precision", 4):
            lines += [f"== {name}", show.to_string(index=False), ""]
    text = "\n".join(lines)
    (OUT.parent / "closing_auction_stdout.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
