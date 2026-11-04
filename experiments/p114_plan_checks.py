"""P114 checks run before the plan was committed (docs/prereg/P114_PERSISTENCE_FIXED_MODEL.md, Section 12).

REAL DATA, reference model only: no extended model is fitted here.

1. The replication's mcsGARCH, refitted through P114's driver code, equals the committed Table 2 values
   (results/replication_nse/table2_mcsgarch.csv) on its three windows and two indices.
2. P114's profile-likelihood code (niftygex.volatility.fit_garch_ext_fixed, through the driver's joint_lr) gives
   the replication's committed likelihood-ratio statistics of equal alpha + beta for the reference model.
3. The one-index grids used for the rolling windows, against the replication's two-index grid.
4. Optional (--p109-committed <folder>): P109's forecasts, recomputed with the shared code of this commit for two
   Nifty refit blocks (--p109-rerun <folder>, made by `p109_mcsgarch_extensions.py run --index NIFTY --blocks 150
   927 --workers 1 --work <folder>`), against the block files of P109's own run, cell by cell as text.

    PYTHONPATH=src OMP_NUM_THREADS=1 python experiments/p114_plan_checks.py [--p109-rerun A --p109-committed B]
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p114", ROOT / "experiments" / "p114_persistence_fixed_model.py")
p114 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p114)
rep = p114.rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--p109-rerun", default=None)
    ap.add_argument("--p109-committed", default=None)
    a = ap.parse_args()
    t0 = time.time()
    print("P114 plan checks (REAL DATA, reference model only; no extended model is fitted)")
    bars = p114.load_bars(Path(a.raw), p114.DATA_END_MAIN)
    G = rep.build_grids(bars, p114.TRIM)
    wins = p114.windows_for(G, False)
    t2 = pd.read_csv(ROOT / "results" / "replication_nse" / "table2_mcsgarch.csv", float_precision="round_trip")
    t2c = pd.read_csv(ROOT / "results" / "replication_nse" / "table2_persistence_change.csv",
                      float_precision="round_trip")
    fits, n_same, n_all = {}, 0, 0
    for w in ("before_stated", "before_inferred", "after"):
        for k in p114.INDICES:
            f = rep.fit_period(G, k, rep.period_rows(G, wins[w]))
            fits[(w, k)] = f
            par = f["mg"].params
            for mine, theirs in (("omega", "omega"), ("alpha", "alpha"), ("beta", "beta"), ("nu", "shape_nu")):
                c = t2[(t2["period"] == w) & (t2["index"] == k) & (t2["parameter"] == theirs)]["ours"].iloc[0]
                n_all += 1
                n_same += int(par[mine] == c)
            print(f"  {w} {k}: {len(wins[w])} days, {f['Rk'].shape[1]} minutes, alpha {par['alpha']!r}, beta {par['beta']!r}")
    print(f"1. Table 2 reference: {n_same} of {n_all} values (omega, alpha, beta, nu) identical to the committed file")
    for w in ("before_inferred", "before_stated"):
        for k in p114.INDICES:
            fb, fa = fits[(w, k)], fits[("after", k)]
            pb, pa = dict(fb["mg"].params), dict(fa["mg"].params)
            r = p114.joint_lr(fb["filt"]["x"], fb["valid_k"].astype(float), pb, fa["filt"]["x"],
                              fa["valid_k"].astype(float), pa, (False, False), "ab", fb["Rk"].shape[1])
            c = t2c[(t2c["before"] == w) & (t2c["index"] == k)].iloc[0]
            print(f"2. LR of equal alpha + beta, reference, {w} {k}: this code {r['lr_stat']:.4f} (common half-life "
                  f"{r['common_half_life']:.1f}), committed {c['lr_equal_persistence_stat']:.4f} (common half-life "
                  f"{c['common_half_life_under_null']:.1f})")
    for k in p114.INDICES:
        Gi = p114.index_grid(bars[k], k)
        extra = Gi["days"].difference(G["days"])
        missing = G["days"].difference(Gi["days"])
        print(f"3. {k}: one-index grid {len(Gi['days'])} days, two-index grid {len(G['days'])} days (to "
              f"{p114.DATA_END_MAIN}); only in the one-index grid: {len(extra)} days "
              f"({', '.join(extra.strftime('%Y-%m-%d')[:3])}{' ...' if len(extra) > 3 else ''}); only in the two-index "
              f"grid: {len(missing)} days")
    if a.p109_rerun and a.p109_committed:
        for b in (150, 927):
            x = pd.read_csv(Path(a.p109_rerun) / f"NIFTY_block{b:04d}.csv", dtype=str)
            y = pd.read_csv(Path(a.p109_committed) / f"NIFTY_block{b:04d}.csv", dtype=str)
            same = x.shape == y.shape and list(x.columns) == list(y.columns)
            n_eq = int((x.values == y.values).sum()) if same else -1
            xp = pd.read_csv(Path(a.p109_rerun) / f"NIFTY_block{b:04d}_params.csv", dtype=str)
            yp = pd.read_csv(Path(a.p109_committed) / f"NIFTY_block{b:04d}_params.csv", dtype=str)
            print(f"4. P109 Nifty block {b}: {len(x)} forecast rows x {x.shape[1]} columns ({', '.join(x.columns[6:])}); "
                  f"identical cells {n_eq} of {x.size}; parameter file identical: {xp.equals(yp)}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
