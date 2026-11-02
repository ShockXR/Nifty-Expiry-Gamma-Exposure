"""Day-level checks on REAL Nifty data (run after scripts/run_pipeline.py --source nse).

Three questions, each answered at the level of days (the unit that is
really independent enough to test; see experiment E2 on why minute-level
tests mislead):

R2  Did expiry-day volatility change under the later expiry rules?
    The paper studies one rule change (Bank Nifty monthly expiry moving to
    Wednesday in 2024). Two more happened later: SEBI's package from
    20 Nov 2024 (one weekly benchmark expiry per exchange, plus larger
    contracts and extra margin on expiry day) and Nifty's move from
    Thursday to Tuesday on 1 Sep 2025. For every week we compute the
    expiry day's log realized variance minus the mean log realized variance
    of the other days of the same week ("week-relative"), which cancels
    volatility shocks common to the whole week, and compare it across the
    three periods.

R3  Does yesterday's dealer gamma (GEX) help explain today's volatility,
    in-sample, beyond yesterday's volatility and VIX? OLS with Newey-West
    standard errors, one regression per sign rule, with an expiry-day
    interaction.

R4  Pinning: on expiry day, does the index close move toward the strike
    with the most open interest more than on other days? We compare the
    change in distance to that strike (close of day t vs close of day t-1,
    same chain) on expiry days and other days, and split expiry days by
    the sign of net GEX under the US convention.

    python experiments/real_day_level.py --run results/pipeline_nse --raw data/raw

Writes results/real_day_level/*.csv and real_day_level_stdout.txt.
Everything here is REAL DATA (NSE files plus validated third-party minute bars).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, wilcoxon

REGIMES = ["multi_weekly", "single_weekly_thu", "single_weekly_tue"]
RULES = {"us_convention": "net_gex_us_convention", "participant_pro": "net_gex_participant",
         "client_mirror": "net_gex_client_mirror"}


def week_relative(dm: pd.DataFrame, col: str) -> pd.DataFrame:
    """Expiry-day log value minus mean log value of the other days in its calendar week."""
    d = dm.copy()
    d["lv"] = np.log(d[col].where(d[col] > 0))
    d["week"] = d.index.to_period("W-SUN")
    rows = []
    for wk, g in d.groupby("week"):
        e = g[g["is_expiry"]]
        o = g[~g["is_expiry"]]
        if len(e) != 1 or len(o) < 2:
            continue
        rows.append({"week": str(wk), "date": e.index[0], "regime": e["regime"].iloc[0],
                     "weekday": int(e["weekday"].iloc[0]), "rel": float(e["lv"].iloc[0] - o["lv"].mean()),
                     "n_other": len(o)})
    return pd.DataFrame(rows)


def r2_regimes(dm: pd.DataFrame) -> pd.DataFrame:
    out = []
    for col in ("rv_day", "rv_first_hour", "rv_last_hour"):
        wr = week_relative(dm, col)
        for reg in REGIMES:
            x = wr.loc[wr["regime"] == reg, "rel"].dropna().to_numpy()
            if len(x) < 5:
                continue
            boot = np.random.default_rng(0).choice(x, (4000, len(x))).mean(axis=1)
            out.append({"measure": col, "regime": reg, "weeks": len(x), "mean_rel": x.mean(),
                        "ratio_exp_vs_week": float(np.exp(x.mean())), "median_rel": float(np.median(x)),
                        "ci95_low": float(np.percentile(boot, 2.5)), "ci95_high": float(np.percentile(boot, 97.5)),
                        "share_weeks_expiry_higher": float((x > 0).mean()),
                        "wilcoxon_p_rel_eq_0": float(wilcoxon(x).pvalue)})
        for a, b in (("multi_weekly", "single_weekly_thu"), ("single_weekly_thu", "single_weekly_tue"),
                     ("multi_weekly", "single_weekly_tue")):
            xa = wr.loc[wr["regime"] == a, "rel"].dropna()
            xb = wr.loc[wr["regime"] == b, "rel"].dropna()
            if len(xa) >= 5 and len(xb) >= 5:
                out.append({"measure": col, "regime": f"{b} minus {a}", "weeks": f"{len(xa)} vs {len(xb)}",
                            "mean_rel": xb.mean() - xa.mean(), "ratio_exp_vs_week": float(np.exp(xb.mean() - xa.mean())),
                            "mannwhitney_p": float(mannwhitneyu(xa, xb).pvalue)})
    return pd.DataFrame(out)


def r3_regressions(dm: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    d = dm.join(panel, how="left")
    d["y"] = np.log(d["rv_day"])
    d["y_l1"] = d["y"].shift(1)
    d["y_w"] = d["y"].shift(1).rolling(5).mean()
    d["lvix_l1"] = np.log(d["vix"]).shift(1)
    d["exp"] = d["is_expiry"].astype(float)
    scale = d["gross_gex"].rolling(20, min_periods=5).median()
    # Regime controls: the participant-based GEX drifts upward over 2023 to 2026 (see the GEX figure),
    # and expiry-day volatility differs by expiry regime, so without these controls a GEX x expiry
    # term could just pick up the regime. Each regime gets its own level and its own expiry effect.
    for reg in REGIMES[1:]:
        d[f"reg_{reg}"] = (d["regime"] == reg).astype(float)
        d[f"exp_x_{reg}"] = d[f"reg_{reg}"] * d["exp"]
    reg_cols = [c for c in d.columns if c.startswith("reg_") or c.startswith("exp_x_")]
    out = []
    specs = {"baseline": []} | {f"gex_{k}": [v] for k, v in RULES.items() if v in d}
    for controls in ("none", "regime x expiry"):
        for name, cols in specs.items():
            X = d[["y_l1", "y_w", "lvix_l1", "exp"]].copy()
            if controls != "none":
                X = X.join(d[reg_cols])
            for c in cols:
                X["gex_l1"] = (d[c] / scale).shift(1)
                X["gex_l1_x_exp"] = X["gex_l1"] * d["exp"]
            ok = X.notna().all(axis=1) & d["y"].notna()
            fit = sm.OLS(d.loc[ok, "y"], sm.add_constant(X[ok])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            for k in fit.params.index:
                out.append({"controls": controls, "spec": name, "term": k, "coef": fit.params[k], "se_hac": fit.bse[k],
                            "t": fit.tvalues[k], "p": fit.pvalues[k], "n_days": int(fit.nobs), "r2": fit.rsquared})
    return pd.DataFrame(out)


def r3_robustness(dm: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """R3 with scale-free GEX measures, gross-gamma controls and a pre-auction sample (review item M5).

    The main R3 scales net GEX by a 20-day median of gross GEX. That ratio has a different spread under
    each sign rule and is mechanically larger on expiry eves (gross gamma peaks one day before expiry),
    which could feed the GEX x expiry term. The variants below remove those channels:

    * per_sd: the main measure divided by its standard deviation within the rule (comparable across rules);
    * sign: a dummy for GEX > 0 (scale-free);
    * share: net GEX / gross GEX of the same chain, in [-1, 1] (no expiry-eve swing in the denominator);
    * share + gross: the share plus log gross GEX and log gross GEX x expiry as controls;
    * share, before auction: the share on days before NSE's closing auction (3 Aug 2026).

    All with regime-specific levels and expiry effects and Newey-West (5 lags) errors. Holm and
    Benjamini-Hochberg adjusted p-values are computed over the whole family of GEX terms in this table.
    """
    from statsmodels.stats.multitest import multipletests
    d = dm.join(panel, how="left")
    d["y"] = np.log(d["rv_day"])
    d["y_l1"] = d["y"].shift(1)
    d["y_w"] = d["y"].shift(1).rolling(5).mean()
    d["lvix_l1"] = np.log(d["vix"]).shift(1)
    d["exp"] = d["is_expiry"].astype(float)
    for reg in REGIMES[1:]:
        d[f"reg_{reg}"] = (d["regime"] == reg).astype(float)
        d[f"exp_x_{reg}"] = d[f"reg_{reg}"] * d["exp"]
    reg_cols = [c for c in d.columns if c.startswith("reg_") or c.startswith("exp_x_")]
    scale = d["gross_gex"].rolling(20, min_periods=5).median()
    d["lgross_l1"] = np.log(d["gross_gex"]).shift(1)
    pre_auction = d.index < pd.Timestamp("2026-08-03")
    rows = []
    for rule, col in RULES.items():
        if col not in d:
            continue
        ratio = (d[col] / scale).shift(1)
        share = (d[col] / d["gross_gex"]).shift(1)
        variants = {
            "per_sd": (ratio / ratio.std(), [], np.ones(len(d), bool)),
            "sign": ((d[col] > 0).astype(float).where(d[col].notna()).shift(1), [], np.ones(len(d), bool)),
            "share": (share, [], np.ones(len(d), bool)),
            "share + gross": (share, ["lgross_l1", "lgross_x_exp"], np.ones(len(d), bool)),
            "share, before auction": (share, [], pre_auction),
        }
        d["lgross_x_exp"] = d["lgross_l1"] * d["exp"]
        for vname, (g, extra, sample) in variants.items():
            X = d[["y_l1", "y_w", "lvix_l1", "exp"] + reg_cols + extra].copy()
            X["gex_l1"] = g
            X["gex_l1_x_exp"] = g * d["exp"]
            ok = X.notna().all(axis=1) & d["y"].notna() & sample
            Xo = X[ok].loc[:, X[ok].std() > 0]
            fit = sm.OLS(d.loc[ok, "y"], sm.add_constant(Xo)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            cov = fit.cov_params()
            tot = fit.params["gex_l1"] + fit.params["gex_l1_x_exp"]
            tot_se = float(np.sqrt(cov.loc["gex_l1", "gex_l1"] + cov.loc["gex_l1_x_exp", "gex_l1_x_exp"]
                                   + 2 * cov.loc["gex_l1", "gex_l1_x_exp"]))
            from scipy.stats import norm
            for term, coef, se in (("gex_l1", fit.params["gex_l1"], fit.bse["gex_l1"]),
                                   ("gex_l1_x_exp", fit.params["gex_l1_x_exp"], fit.bse["gex_l1_x_exp"]),
                                   ("total on expiry days", tot, tot_se)):
                rows.append({"rule": rule, "variant": vname, "term": term, "coef": float(coef), "se_hac": float(se),
                             "p": float(2 * (1 - norm.cdf(abs(coef / se)))), "n_days": int(fit.nobs)})
    out = pd.DataFrame(rows)
    fam = out["term"] != "total on expiry days"
    out.loc[fam, "p_holm"] = multipletests(out.loc[fam, "p"], method="holm")[1]
    out.loc[fam, "p_bh"] = multipletests(out.loc[fam, "p"], method="fdr_bh")[1]
    return out


def r4_pinning(dm: pd.DataFrame, panel: pd.DataFrame, official_close: pd.Series) -> pd.DataFrame:
    d = dm.join(panel[["max_oi_strike", "max_pain", "net_gex_us_convention", "gross_gex"]].shift(1), how="left")
    d["close"] = official_close.reindex(d.index)
    d["close_prev"] = d["close"].shift(1)
    out = []
    for target in ("max_oi_strike", "max_pain"):
        K = d[target]
        dist_now = (d["close"] / K - 1).abs()
        dist_prev = (d["close_prev"] / K - 1).abs()
        change = (dist_now - dist_prev) * 1e4  # basis points; negative = moved toward the strike
        ok = change.notna()
        e, o = change[ok & d["is_expiry"]], change[ok & ~d["is_expiry"]]
        pos = ok & d["is_expiry"] & (d["net_gex_us_convention"] > 0)
        neg = ok & d["is_expiry"] & (d["net_gex_us_convention"] <= 0)
        for label, x in (("expiry days", e), ("other days", o), ("expiry, US-sign GEX > 0", change[pos]),
                         ("expiry, US-sign GEX <= 0", change[neg])):
            out.append({"target": target, "group": label, "n": len(x), "mean_change_bp": x.mean(),
                        "median_change_bp": x.median(), "share_moved_toward": float((x < 0).mean()),
                        "median_dist_after_bp": float((dist_now[x.index] * 1e4).median())})
        out.append({"target": target, "group": "expiry minus other", "n": f"{len(e)} vs {len(o)}",
                    "mean_change_bp": e.mean() - o.mean(), "median_change_bp": e.median() - o.median(),
                    "mannwhitney_p": float(mannwhitneyu(e, o).pvalue)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/pipeline_nse")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/real_day_level")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dm = pd.read_csv(run / "daily_market.csv", index_col=0, parse_dates=True)
    panel = pd.read_csv(run / "daily_gex_panel.csv", index_col=0, parse_dates=True)
    idx = pd.read_csv(Path(a.raw) / "index_daily.csv", parse_dates=["date"])
    official = idx[idx["index"].str.lower() == "nifty 50"].set_index("date")["close"]

    lines = ["REAL DATA. Nifty 50, days " + f"{dm.index.min().date()} to {dm.index.max().date()} ({len(dm)} days, "
             f"{int(dm['is_expiry'].sum())} expiry days)."]
    lines.append("Days per regime: " + ", ".join(f"{k} {v}" for k, v in dm["regime"].value_counts().items()))
    lines.append("Expiry weekday counts (0=Mon): " + str(dm.loc[dm["is_expiry"], "weekday"].value_counts().sort_index().to_dict()))
    for name, df in (("r2_expiry_effect_by_regime", r2_regimes(dm)),
                     ("r3_gex_regressions", r3_regressions(dm, panel)),
                     ("r3_robustness", r3_robustness(dm, panel)),
                     ("r4_pinning", r4_pinning(dm, panel, official))):
        df.to_csv(out / f"{name}.csv", index=False)
        with pd.option_context("display.width", 220, "display.max_columns", 20, "display.precision", 4):
            lines += ["", f"== {name}", df.to_string(index=False)]
    text = "\n".join(lines)
    (out.parent / "real_day_level_stdout.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
