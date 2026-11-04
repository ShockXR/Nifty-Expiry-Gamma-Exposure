#!/usr/bin/env python
"""Redraw the pipeline figures (scoreboard, ablation, GEX by sign rule) from the COMMITTED CSV files.

Nothing is refitted and no raw data is needed. The titles are the ones scripts/run_pipeline.py uses; the
only change is what niftygex.plots now adds (data label, number of test days, colour meaning).
The diurnal curve cannot be redrawn this way: its curve is not saved in any committed file.

    PYTHONPATH=src python reports/figure_regen/regen_pipeline_figures.py [--out-root DIR]

Without --out-root the PNGs are rewritten in place (same file names). With it they go to DIR/<run>/figures/.
Written by an AI agent (Claude) for Shreyan Thanki's project.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from niftygex import plots

ROOT = Path(__file__).resolve().parents[2]
RUNS = {  # folder -> (title prefix, data source)
    "pipeline_nse": ("", "nse"),
    "pipeline_nse_v2": ("", "nse"),
    "pipeline_nse_banknifty": ("BANKNIFTY: ", "nse"),
    "pipeline_nse_finnifty": ("FINNIFTY: ", "nse"),
    "pipeline_synthetic": ("", "synthetic"),
}
TARGETS = ("next15", "next30", "last_hour")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args()
    done = []
    for run, (mk, src) in RUNS.items():
        rd = ROOT / "results" / run
        if not rd.exists():
            continue
        fig_dir = (Path(a.out_root) / run / "figures") if a.out_root else rd / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        label = "REAL DATA (NSE)" if src == "nse" else "SYNTHETIC (simulated market, not real data)"
        gp = rd / "daily_gex_panel.csv"
        if gp.exists():
            panel = pd.read_csv(gp, index_col=0, parse_dates=True)
            plots.gex_timeseries(panel, fig_dir / "gex_by_sign_rule.png",
                                 f"{mk}Net GEX / {plots.GEX_SCALE_WINDOW}-day median gross GEX, by sign rule "
                                 f"({'synthetic' if src == 'synthetic' else 'NSE'})", data_label=label)
            done.append(f"{run}/gex_by_sign_rule")
        for name in TARGETS:
            sbp = rd / f"scoreboard_{name}.csv"
            if sbp.exists():
                sb = pd.read_csv(sbp)
                for g in ("all", "expiry"):
                    plots.scoreboard_bars(sb, g, fig_dir / f"scoreboard_{name}_{g}.png",
                                          f"{mk}{name}, {g} days: out-of-sample QLIKE vs mcsGARCH", data_label=label)
                    done.append(f"{run}/scoreboard_{name}_{g}")
            ap_ = rd / f"ablation_{name}.csv"
            if ap_.exists():
                abl = pd.read_csv(ap_)
                for kind in ("hybrid", "hybridiv", "linear"):
                    plots.ablation_dots(abl, f"{kind}: base vs base+GEX", fig_dir / f"ablation_{name}_{kind}.png",
                                        f"{mk}{name}: {kind} model, adding each GEX set (REAL DATA)" if src == "nse"
                                        else f"{name}: {kind} model, adding each GEX set (synthetic)", data_label=label)
                    done.append(f"{run}/ablation_{name}_{kind}")
    print(f"redrew {len(done)} figures")


if __name__ == "__main__":
    main()
