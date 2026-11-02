"""R5 (alternative A5 as a working prototype): end-of-day momentum conditioned on dealer gamma. REAL DATA.

Baltussen, Da, Lammers and Martens (2021, JFE) show that in many futures
markets the return over most of the day predicts the return over the last 30
minutes ("market intraday momentum", Gao, Han, Li and Zhou 2018, JFE), and
that the effect is stronger when option dealers are short gamma: dealers who
are short gamma must buy after the market rises and sell after it falls, and
much of that hedging happens near the close. This script asks whether the
same holds for Nifty, using the GEX that was known the evening before.

For each day t (1-minute Nifty bars, IST):

    r_rod   = log(price at 14:59 bar close / previous day's last bar close)
              (the "rest of day" return, overnight gap included, as in Baltussen et al.)
    r_last  = log(15:29 bar close / 14:59 bar close)   (the last 30 minutes)

    r_last = a + b r_rod + c r_rod * 1[GEX_{t-1} < 0] + d 1[GEX_{t-1} < 0] + e

with Newey-West standard errors (5 lags), on all days, non-expiry days and
expiry days, for each of the three sign rules. The hedging story predicts
c > 0. A second specification lets the momentum slope b differ by expiry
regime, so that a slow drift in GEX (the participant-based measure trends
upward over 2023 to 2026) cannot masquerade as a gamma effect. A third uses
GEX as a continuous variable. A fourth adds yesterday's India VIX and the size
of today's move as competing explanations (reversal may simply be stronger
in a busy, volatile market, whatever dealers hold).

A by-regime and first-half / second-half split of the binary interaction is
written to results/momentum_gex/stability.csv.

    python experiments/momentum_gex.py --run results/pipeline_nse --raw data/raw

Writes results/momentum_gex/*.csv and results/momentum_gex_stdout.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

RULES = {"us_convention": "net_gex_us_convention", "participant_pro": "net_gex_participant",
         "client_mirror": "net_gex_client_mirror"}
SLOT_1459, SLOT_1529 = 344, 374  # minutes after 09:15


def daily_returns(bars_path: Path, days: pd.DatetimeIndex) -> pd.DataFrame:
    b = pd.read_csv(bars_path, parse_dates=["timestamp"])
    b = b[(b["timestamp"] >= days.min() - pd.Timedelta(days=7)) & (b["timestamp"] < days.max() + pd.Timedelta(days=1))]
    b["day"] = b["timestamp"].dt.normalize()
    b["slot"] = ((b["timestamp"] - b["day"]) / pd.Timedelta(minutes=1) - 555).astype(int)  # 09:15 = 555 minutes
    piv = b.pivot_table(index="day", columns="slot", values="close", aggfunc="last")
    piv = piv.reindex(days)
    last = piv.get(SLOT_1529)
    p1459 = piv.get(SLOT_1459)
    prev_last = last.shift(1)
    out = pd.DataFrame({"p_prev_close": prev_last, "p_1459": p1459, "p_1529": last}, index=days)
    out["r_rod"] = np.log(out["p_1459"] / out["p_prev_close"])
    out["r_last"] = np.log(out["p_1529"] / out["p_1459"])
    return out


def fit(y, X, lags=5):
    ok = X.notna().all(axis=1) & y.notna()
    return sm.OLS(y[ok], sm.add_constant(X[ok])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/pipeline_nse")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results/momentum_gex")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dm = pd.read_csv(Path(a.run) / "daily_market.csv", index_col=0, parse_dates=True)
    panel = pd.read_csv(Path(a.run) / "daily_gex_panel.csv", index_col=0, parse_dates=True)
    d = dm.join(daily_returns(Path(a.raw) / "nifty_1min.csv", dm.index))
    # The gap between consecutive trading days must be a real overnight (the previous row is the previous trading day).
    scale = panel["gross_gex"].rolling(20, min_periods=5).median()
    for k, col in RULES.items():
        d[f"gex_{k}"] = (panel[col] / scale).shift(1).reindex(d.index)  # known at the previous close
    # Controls for the "busy market" explanation: yesterday's log VIX (demeaned) and the size of today's move.
    # The shift is taken on the full trading calendar, before any day is dropped, so "yesterday" is always
    # the previous trading day even when that day has no usable minute bars.
    d["lvix_prev"] = np.log(d["vix"]).shift(1)
    d = d.dropna(subset=["r_rod", "r_last"])
    d["r_rod_bp"], d["r_last_bp"] = d["r_rod"] * 1e4, d["r_last"] * 1e4
    d["lvix_prev"] -= d["lvix_prev"].mean()

    rows, summ = [], []
    subsets = {"all": np.ones(len(d), bool), "non_expiry": ~d["is_expiry"].to_numpy(bool),
               "expiry": d["is_expiry"].to_numpy(bool)}
    # Baseline: plain intraday momentum
    for sname, m in subsets.items():
        f = fit(d.loc[m, "r_last_bp"], d.loc[m, ["r_rod_bp"]])
        rows.append({"spec": "momentum only", "rule": "none", "subset": sname, "term": "r_rod", "coef": f.params["r_rod_bp"],
                     "se_hac": f.bse["r_rod_bp"], "p": f.pvalues["r_rod_bp"], "n": int(f.nobs), "r2": f.rsquared})
    for k in RULES:
        g = d[f"gex_{k}"]
        neg = (g < 0).astype(float).where(g.notna())  # a day with no GEX is missing, not "non-negative"
        for sname, m in subsets.items():
            dd = d[m]
            X = pd.DataFrame({"r_rod": dd["r_rod_bp"], "neg": neg[m], "r_rod_x_neg": dd["r_rod_bp"] * neg[m]})
            specs = {"binary": X}
            Xr = X.copy()
            for reg in ("single_weekly_thu", "single_weekly_tue"):
                ind = (dd["regime"] == reg).astype(float)
                Xr[f"reg_{reg}"] = ind
                Xr[f"r_rod_x_{reg}"] = dd["r_rod_bp"] * ind
            specs["binary + regime slopes"] = Xr
            Xc = X.copy()
            Xc["lvix_prev"] = dd["lvix_prev"]
            Xc["r_rod_x_lvix"] = dd["r_rod_bp"] * dd["lvix_prev"]
            Xc["r_rod_x_abs_r_rod"] = dd["r_rod_bp"] * dd["r_rod_bp"].abs()
            specs["binary + VIX and size controls"] = Xc
            specs["continuous"] = pd.DataFrame({"r_rod": dd["r_rod_bp"], "gex": g[m], "r_rod_x_gex": dd["r_rod_bp"] * g[m]})
            for spec, XX in specs.items():
                if XX.get("neg") is not None and XX["neg"].nunique() < 2:
                    continue
                f = fit(dd["r_last_bp"], XX)
                for term in f.params.index:
                    rows.append({"spec": spec, "rule": k, "subset": sname, "term": term, "coef": f.params[term],
                                 "se_hac": f.bse[term], "p": f.pvalues[term], "n": int(f.nobs), "r2": f.rsquared})
            # descriptive: how often each gamma state occurs, and the "follow the day's direction" payoff
            for state, mm in (("GEX < 0", neg[m] == 1), ("GEX >= 0", neg[m] == 0)):
                x = dd[mm.to_numpy(bool)]
                pay = np.sign(x["r_rod_bp"]) * x["r_last_bp"]
                summ.append({"rule": k, "subset": sname, "state": state, "days": len(x),
                             "mean_follow_payoff_bp": pay.mean(), "t_stat": pay.mean() / (pay.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan,
                             "corr_rod_last": x[["r_rod_bp", "r_last_bp"]].corr().iloc[0, 1] if len(x) > 2 else np.nan})
    stab = []
    for k in RULES:
        neg = (d[f"gex_{k}"] < 0).astype(float).where(d[f"gex_{k}"].notna())
        half = d.index < d.index[len(d) // 2]
        parts = [(f"regime {r}", (d["regime"] == r).to_numpy()) for r in d["regime"].unique()]
        parts += [("first half", half), ("second half", ~half)]
        for name, m in parts:
            dd, nn = d[m], neg[m]
            if nn.nunique() < 2:
                stab.append({"rule": k, "part": name, "days": int(m.sum()), "neg_days": int(nn.sum())})
                continue
            f = fit(dd["r_last_bp"], pd.DataFrame({"r_rod": dd["r_rod_bp"], "neg": nn, "r_rod_x_neg": dd["r_rod_bp"] * nn}))
            stab.append({"rule": k, "part": name, "days": int(f.nobs), "neg_days": int(nn.sum()),
                         "r_rod": f.params["r_rod"], "r_rod_x_neg": f.params["r_rod_x_neg"], "p": f.pvalues["r_rod_x_neg"]})
    st = pd.DataFrame(stab)
    st.to_csv(out / "stability.csv", index=False)
    res, sm_ = pd.DataFrame(rows), pd.DataFrame(summ)
    # Multiple testing: Holm and Benjamini-Hochberg over the whole family of GEX interaction tests
    # (binary r_rod_x_neg and continuous r_rod_x_gex terms, every rule, specification and day subset).
    from statsmodels.stats.multitest import multipletests
    fam = res["term"].isin(["r_rod_x_neg", "r_rod_x_gex"])
    res.loc[fam, "p_holm_family"] = multipletests(res.loc[fam, "p"], method="holm")[1]
    res.loc[fam, "p_bh_family"] = multipletests(res.loc[fam, "p"], method="fdr_bh")[1]
    res.loc[fam, "family_size"] = int(fam.sum())
    res.to_csv(out / "regressions.csv", index=False)
    sm_.to_csv(out / "gamma_states.csv", index=False)
    key = res[res["term"].isin(["r_rod", "r_rod_x_neg", "r_rod_x_gex", "r_rod_x_lvix", "r_rod_x_abs_r_rod"])]
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.precision", 4):
        text = "\n".join(["REAL DATA. Nifty 50 end-of-day momentum conditioned on GEX known the evening before.",
                          f"Days: {len(d)} ({int(d['is_expiry'].sum())} expiry). Returns in basis points.", "",
                          "== key coefficients", key.to_string(index=False), "", "== gamma states (descriptive; "
                          "'follow payoff' = sign(rest-of-day return) x last-30-minute return, before costs, not a trading claim)",
                          sm_.to_string(index=False), "", "== stability of the binary interaction", st.to_string(index=False)])
    (out.parent / "momentum_gex_stdout.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
