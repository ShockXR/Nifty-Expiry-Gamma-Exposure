"""A2b. Does the EGARCH-versus-EWMA choice in niftygex.volatility.daily_egarch_variance depend on days AFTER the refit?
Uses the committed REAL daily panel (results/pipeline_nse_v2/daily_market.csv): daily return ~ log(close_last_bar / open).
That is an approximation of the pipeline's R.sum(axis=1), so the result is an OUR CHECK, not a re-run of the committed pipeline.
For every refit start s = 150, 171, ... compare the method chosen with (a) the whole series, (b) data up to s + 21, (c) data up to s only (strictly causal)."""
import sys, numpy as np, pandas as pd
from niftygex.volatility import daily_egarch_variance
d = pd.read_csv(sys.argv[1] + "/results/pipeline_nse_v2/daily_market.csv")
y = np.log(d.close_last_bar / d.open).to_numpy(); T = len(y)
rows = []
for s in range(150, T, 21):
    ha, ia = daily_egarch_variance(y, train_size=s)
    hb, ib = daily_egarch_variance(y[:min(s + 21, T)], train_size=s)
    hc, ic = daily_egarch_variance(y[:s + 1], train_size=s)
    rows.append(dict(start=s, method_full=ia["method"], method_to_s_plus_21=ib["method"], method_to_s=ic["method"],
                     max_rel_diff_h_full_vs_s_plus_21_over_block=float(np.max(np.abs(ha[s:s + 21] - hb[s:min(s + 21, T)][:len(ha[s:s + 21])]) / ha[s:s + 21])) if len(hb) > s else np.nan))
r = pd.DataFrame(rows); r.to_csv(sys.argv[2], index=False)
print(r[["method_full", "method_to_s_plus_21", "method_to_s"]].apply(pd.Series.value_counts).fillna(0).astype(int).to_string())
print("starts where full-series choice differs from the choice made with data up to s + 21:", r.loc[r.method_full != r.method_to_s_plus_21, "start"].tolist())
print("starts where it differs from the strictly causal choice:", r.loc[r.method_full != r.method_to_s, "start"].tolist())
print("max relative difference of h inside the block:", r.max_rel_diff_h_full_vs_s_plus_21_over_block.max())
