"""Regenerate every real-data table and figure with one command.

    python scripts/reproduce_all.py                 # data already in data/raw
    python scripts/reproduce_all.py --download      # fetch the data first (about 1 to 2 hours, polite to NSE)
    python scripts/reproduce_all.py --only replicate_paper pinning_study

Steps, in order:

1. (with --download) NSE files for 2 Jan 2023 to 2 Nov 2026 and, for the
   paper replication, 20 Oct 2022 to 30 Dec 2022 into data/raw/extra_2022;
   minute bars from the two GitHub datasets at the pinned commits.
2. Check every raw file against data/MANIFEST.csv (SHA-256). The run stops
   if anything is missing or different, unless --no-check is given.
3. Run each study in turn, with fixed arguments, and log its run time and
   exit status to results/reproduce_log.csv.

All results are REAL DATA except the simulation experiments (E1 to E7),
which have their own commands in docs/REPRODUCIBILITY.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PY = sys.executable
END = "2026-11-02"
SOURCES = json.loads(Path("data/raw/minute_bars_source.json").read_text()) if Path("data/raw/minute_bars_source.json").exists() else {
    "nifty": {"commit": "7f94ced8465c7b824d7c7b5f92f5a1151a8d04be"},
    "banknifty": {"commit": "6bdda6e571dd5e345ef3441df528a1fdccdbb0e9"}}

DOWNLOAD = [
    ("download_nse", [PY, "scripts/download_nse.py", "--start", "2023-01-01", "--end", END, "--out", "data/raw"]),
    ("download_nse_2022", [PY, "scripts/download_nse.py", "--start", "2022-10-20", "--end", "2022-12-30", "--out", "data/raw/extra_2022"]),
    ("fetch_minute_bars", [PY, "scripts/fetch_minute_bars.py", "--out", "data/raw", "--until", END,
                           "--nifty-ref", SOURCES["nifty"]["commit"], "--bank-ref", SOURCES["banknifty"]["commit"]]),
    ("fetch_sensex_bars", [PY, "experiments/sensex_confound.py", "--fetch-sensex", "--fetch-only", "--until", END]),
]

# name, command, outputs (for the log only)
STEPS = [
    ("validate_minute_bars", [PY, "scripts/validate_minute_bars.py"], "results/data_validation/"),
    ("check_oi_units", [PY, "scripts/check_oi_units.py"], "results/data_validation/"),
    # The forecasting contest (v2: mcsGARCH refit every 21 days, option-implied features, 5 LightGBM seeds).
    # About 100 minutes on one core. Every later study reads its daily panels from this folder.
    ("run_pipeline", [PY, "scripts/run_pipeline.py", "--source", "nse", "--end", END, "--n-seeds", "5", "--seed-study", "3",
                      "--out", "results/pipeline_nse_v2"], "results/pipeline_nse_v2/"),
    ("forecast_diagnostics", [PY, "experiments/forecast_diagnostics.py"], "results/forecast_diagnostics/"),
    ("real_day_level", [PY, "experiments/real_day_level.py"], "results/real_day_level/"),
    ("momentum_gex", [PY, "experiments/momentum_gex.py"], "results/momentum_gex/"),
    ("closing_auction_check", [PY, "experiments/closing_auction_check.py"], "results/closing_auction/"),
    ("replicate_paper", [PY, "experiments/replicate_paper.py"], "results/replication_nse/"),
    ("pinning_study", [PY, "experiments/pinning_study.py"], "results/pinning/"),
    ("notebook", ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "notebooks/01_real_data_tour.ipynb"],
     "notebooks/01_real_data_tour.ipynb"),
]
OPTIONAL = {  # studies added later; run if the script exists
    "falsification": ([PY, "experiments/falsification.py"], "results/falsification/"),
    "intraday_gamma": ([PY, "experiments/intraday_gamma.py"], "results/intraday_gamma/"),
    "dealer_sign": ([PY, "experiments/dealer_sign.py"], "results/dealer_sign/"),
    "expiry_event_study": ([PY, "experiments/expiry_event_study.py"], "results/event_study/"),
    "sensex_confound": ([PY, "experiments/sensex_confound.py"], "results/sensex_confound/"),
    "connectedness_gamma": ([PY, "experiments/connectedness_gamma.py"], "results/connectedness_gamma/"),
    "power_real": ([PY, "experiments/power_real.py"], "results/power_real/"),
    "rekey_check": ([PY, "experiments/rekey_check.py"], "results/replication_nse/rekey_check.csv"),
    "pinning_mechanism": ([PY, "experiments/pinning_mechanism.py"], "results/pinning_mechanism/"),
    "f2_economics": ([PY, "experiments/f2_economics.py"], "results/f2_economics/"),
    "sensex_swap": ([PY, "experiments/sensex_swap.py"], "results/sensex_swap/"),
}


def run(name: str, cmd: list[str], env: dict) -> dict:
    t0 = time.time()
    print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, env=env)
    out = {"step": name, "command": " ".join(cmd), "exit_code": r.returncode, "seconds": round(time.time() - t0, 1)}
    print(f"=== {name}: exit {r.returncode} in {out['seconds']} s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--no-check", action="store_true", help="skip the manifest check (not recommended)")
    ap.add_argument("--only", nargs="+", default=None, help="run only these steps")
    a = ap.parse_args()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
    log = []
    if a.download:
        for name, cmd in DOWNLOAD:
            log.append(run(name, cmd, env))
    if not a.no_check:
        res = run("data_manifest_check", [PY, "scripts/data_manifest.py", "check"], env)
        log.append(res)
        if res["exit_code"] != 0:
            print("Data do not match data/MANIFEST.csv; stopping. Use --download, or --no-check to run anyway.")
            pd.DataFrame(log).to_csv("results/reproduce_log.csv", index=False)
            return 1
    steps = list(STEPS) + [(k, c, o) for k, (c, o) in OPTIONAL.items() if Path(c[1]).exists()]
    if a.only:
        steps = [s for s in steps if s[0] in set(a.only)]
    for name, cmd, outputs in steps:
        res = run(name, cmd, env)
        res["outputs"] = outputs
        log.append(res)
    df = pd.DataFrame(log)
    df.to_csv("results/reproduce_log.csv", index=False)
    print("\n" + df[["step", "exit_code", "seconds"]].to_string(index=False))
    return int((df["exit_code"] != 0).any())


if __name__ == "__main__":
    sys.exit(main())
