"""Regenerate every real-data table and figure with one command.

    python scripts/reproduce_all.py                 # data already in data/raw
    python scripts/reproduce_all.py --download      # fetch the data first (about 1 to 2 hours, polite to NSE)
    python scripts/reproduce_all.py --only replicate_paper pinning_study
    python scripts/reproduce_all.py --list          # print every step and its command; runs nothing

Steps, in order:

1. (with --download) NSE files for 2 Jan 2023 to 2 Nov 2026 and, for the
   paper replication, 20 Oct 2022 to 30 Dec 2022 into data/raw/extra_2022;
   minute bars from the two GitHub datasets at the pinned commits.
2. Check every raw file against data/MANIFEST.csv (SHA-256). The run stops
   if anything is missing or different, unless --no-check is given.
3. Run each study in turn, with fixed arguments, and log its run time and
   exit status to results/reproduce_log.csv. Each registered study (P87 to
   P118) runs with the commands given in its document; the test ledger and the
   number trace run last, because they read every study's results.

Every step runs from the repository root with OMP_NUM_THREADS=1, PYTHONHASHSEED=0
and the repository's src/ first on PYTHONPATH, as the study documents' commands do.
Run times stated in the comments below come from the study documents; see
docs/REPRODUCIBILITY.md for which steps have been timed in one end-to-end run.

All results are REAL DATA except the simulation experiments (E1 to E7),
which have their own commands in docs/REPRODUCIBILITY.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PY = sys.executable
END = "2026-11-02"
SRC = Path(__file__).resolve().parents[1] / "src"
SOURCES = json.loads(Path("data/raw/minute_bars_source.json").read_text()) if Path("data/raw/minute_bars_source.json").exists() else {
    "nifty": {"commit": "7f94ced8465c7b824d7c7b5f92f5a1151a8d04be"},
    "banknifty": {"commit": "6bdda6e571dd5e345ef3441df528a1fdccdbb0e9"}}

DOWNLOAD = [
    ("download_nse", [PY, "scripts/download_nse.py", "--start", "2023-01-01", "--end", END, "--out", "data/raw"]),
    ("download_nse_2022", [PY, "scripts/download_nse.py", "--start", "2022-10-20", "--end", "2022-12-30", "--out", "data/raw/extra_2022"]),
    ("fetch_minute_bars", [PY, "scripts/fetch_minute_bars.py", "--out", "data/raw", "--until", END,
                           "--nifty-ref", SOURCES["nifty"]["commit"], "--bank-ref", SOURCES["banknifty"]["commit"]]),
    ("fetch_sensex_bars", [PY, "experiments/sensex_confound.py", "--fetch-sensex", "--fetch-only", "--until", END]),
    # NSE's full F&O bhavcopies and all-index closes for the synthetic control (P96), 2022 to 2026
    ("fetch_nse_full", [PY, "experiments/synthetic_control.py", "fetch", "--out", "data/raw/nse_full"]),
    # Fin Nifty bars (pinned commit and SHA-256), official days and bhavcopies for P104; needs nse_full first
    ("fetch_finnifty_data", [PY, "scripts/fetch_finnifty_data.py", "--raw", "data/raw"]),
]

# name, command, outputs (for the log only)
STEPS = [
    ("validate_minute_bars", [PY, "scripts/validate_minute_bars.py"], "results/data_validation/"),
    ("check_oi_units", [PY, "scripts/check_oi_units.py"], "results/data_validation/"),
    # The forecasting contest (v2: mcsGARCH refit every 21 days, option-implied features, 5 LightGBM seeds).
    # About 100 to 115 minutes on one core. Every later study reads its daily panels from this folder.
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
# name: (command, outputs[, file for the step's stdout and stderr])
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
    # Added on 3 November 2026 after the timed run: the synthetic control (P96), the Bank Nifty forecasting replication (P91,
    # about 80 minutes), the Fin Nifty replication (P104), the referee-response analyses (post hoc), the test
    # ledger and the number trace.
    "synthetic_control": ([PY, "experiments/synthetic_control.py", "analyse", "--phase2", "--full", "data/raw/nse_full"],
                          "results/synthetic_control/"),
    "banknifty_pipeline": ([PY, "scripts/run_pipeline.py", "--source", "nse", "--symbol", "BANKNIFTY", "--end", "2026-04-22",
                            "--n-seeds", "5", "--seed-study", "3", "--verbose", "--out", "results/pipeline_nse_banknifty"],
                           "results/pipeline_nse_banknifty/", "results/pipeline_nse_banknifty_stdout.txt"),
    "p91_banknifty_summary": ([PY, "experiments/p91_banknifty_summary.py"], "results/pipeline_nse_banknifty/p91/"),
    # The third-index replication on Fin Nifty (P104, about 70 minutes on one core).
    "validate_finnifty_bars": ([PY, "scripts/validate_minute_bars.py", "--indices", "Nifty Financial Services",
                                "--out", "results/data_validation_finnifty"], "results/data_validation_finnifty/"),
    "finnifty_pipeline": ([PY, "scripts/run_pipeline.py", "--source", "nse", "--symbol", "FINNIFTY", "--end", "2026-01-27",
                           "--n-seeds", "5", "--seed-study", "3", "--verbose", "--out", "results/pipeline_nse_finnifty"],
                          "results/pipeline_nse_finnifty/", "results/pipeline_nse_finnifty_stdout.txt"),
    "p104_finnifty_summary": ([PY, "experiments/p104_finnifty_summary.py"], "results/pipeline_nse_finnifty/p104/"),
    "p104_finnifty_exploratory": ([PY, "experiments/p104_finnifty_exploratory.py"],
                                  "results/pipeline_nse_finnifty/p104/exploratory.csv"),
    "referee_forecast": ([PY, "experiments/referee_forecast.py"], "results/referee_forecast/",
                         "results/referee_forecast_stdout.txt"),
    "referee_forecast_gw": ([PY, "experiments/referee_forecast_gw.py"], "results/referee_forecast/",
                            "results/referee_forecast_gw_stdout.txt"),
    "referee_f2": ([PY, "experiments/referee_f2.py"], "results/referee_f2/"),
    "referee_f2_positions": ([PY, "experiments/referee_f2_positions.py"], "results/referee_f2/"),
    "referee_expiry": ([PY, "experiments/referee_expiry.py"], "results/referee_expiry/"),
    "f2_dsr_power": ([PY, "experiments/f2_dsr_power.py"], "results/f2_economics/dsr_power_posthoc.csv",
                     "results/f2_economics/dsr_power_posthoc_stdout.txt"),
    "p103_economic_value": ([PY, "experiments/p103_economic_value.py"], "results/p103_economic_value/"),
    # ---- Registered follow-up studies, added 3 November 2026 with the commands of each study's document (in
    # dependency order). These steps have not yet been timed in one end-to-end run; the times are those stated in
    # the documents.
    # P99 seed bootstrap (docs/SEED_BOOTSTRAP.md, Section 7): about 71 minutes with 2 worker processes.
    # Needs results/pipeline_nse_v2 and, for the post hoc context step, results/referee_forecast.
    "p99_seed_bootstrap_run": ([PY, "experiments/seed_bootstrap.py", "run", "--raw", "data/raw", "--workers", "2"],
                               "results/seed_bootstrap/", "results/seed_bootstrap/run_stdout.txt"),
    "p99_seed_bootstrap_analyze": ([PY, "experiments/seed_bootstrap.py", "analyze",
                                    "--plan-commit", "64c6151314bd532162a54219069d08a2e4597cb3"],
                                   "results/seed_bootstrap/", "results/seed_bootstrap/analyze_stdout.txt"),
    "p99_seed_bootstrap_context": ([PY, "experiments/seed_bootstrap.py", "context"],
                                   "results/seed_bootstrap/m43_comparison.csv", "results/seed_bootstrap/context_stdout.txt"),
    # P106 gain decomposition (docs/P106_GAIN_DECOMPOSITION.md, Section 10; commands of its plan, Section 5):
    # about 1 hour 45 minutes on 2 cores. Needs the three pipeline runs and P99.
    "p106_run_nifty": ([PY, "experiments/p106_gain_decomposition.py", "run", "--index", "NIFTY", "--workers", "2"],
                       "results/p106_gain_decomposition/", "results/p106_gain_decomposition/run_stdout_nifty.txt"),
    "p106_run_banknifty": ([PY, "experiments/p106_gain_decomposition.py", "run", "--index", "BANKNIFTY", "--workers", "2"],
                           "results/p106_gain_decomposition/", "results/p106_gain_decomposition/run_stdout_banknifty.txt"),
    "p106_run_finnifty": ([PY, "experiments/p106_gain_decomposition.py", "run", "--index", "FINNIFTY", "--workers", "2"],
                          "results/p106_gain_decomposition/", "results/p106_gain_decomposition/run_stdout_finnifty.txt"),
    "p106_analyze": ([PY, "experiments/p106_gain_decomposition.py", "analyze",
                      "--plan-commit", "71bb3b6b922540021082a40556cf8d915a45f085"],
                     "results/p106_gain_decomposition/", "results/p106_gain_decomposition/analyze_stdout.txt"),
    # P107 pooled three-index test (docs/P107_POOLED_THREE_INDEX.md, Section 12): about 15 seconds on one core.
    "p107_count": ([PY, "experiments/p107_pooled_test.py", "count"], "results/p107_pooled_test/phase1/"),
    "p107_run": ([PY, "experiments/p107_pooled_test.py", "run", "--plan-commit", "eb8580af15d0ddbd6e875887440a2d73fbd663c4"],
                 "results/p107_pooled_test/"),
    "p107_posthoc": ([PY, "experiments/p107_posthoc.py"], "results/p107_pooled_test/posthoc_influence.csv"),
    # P108 blind window search (docs/P108_BLIND_WINDOW.md, Section 9): about three minutes. Raw data only.
    "p108_search": ([PY, "experiments/p108_blind_window/search.py", "--raw", "data/raw", "--out", "results/p108_blind_window",
                     "--plan-commit", "6b289ca7b3f29803dfab0159377235eb7452761b"], "results/p108_blind_window/"),
    "p108_posthoc": ([PY, "experiments/p108_blind_window/posthoc.py", "--raw", "data/raw", "--out", "results/p108_blind_window"],
                     "results/p108_blind_window/"),
    "p108_phase3_compare": ([PY, "experiments/p108_blind_window/phase3_compare.py", "--raw", "data/raw",
                             "--out", "results/p108_blind_window"], "results/p108_blind_window/"),
    # P109 mcsGARCH extensions (docs/P109_MCSGARCH_EXTENSIONS.md, Section 17): about 16 minutes on 2 cores.
    # Needs P99 and P106.
    "p109_run_nifty": ([PY, "experiments/p109_mcsgarch_extensions.py", "run", "--index", "NIFTY", "--workers", "2"],
                       "results/p109_mcsgarch_extensions/", "results/p109_mcsgarch_extensions/run_stdout_nifty.txt"),
    "p109_run_banknifty": ([PY, "experiments/p109_mcsgarch_extensions.py", "run", "--index", "BANKNIFTY", "--workers", "2"],
                           "results/p109_mcsgarch_extensions/", "results/p109_mcsgarch_extensions/run_stdout_banknifty.txt"),
    "p109_run_finnifty": ([PY, "experiments/p109_mcsgarch_extensions.py", "run", "--index", "FINNIFTY", "--workers", "2"],
                          "results/p109_mcsgarch_extensions/", "results/p109_mcsgarch_extensions/run_stdout_finnifty.txt"),
    "p109_analyze": ([PY, "experiments/p109_mcsgarch_extensions.py", "analyze",
                      "--plan-commit", "ceec2ec2d4287c1fa0a2616dfe8eac137fb2c3dd"],
                     "results/p109_mcsgarch_extensions/", "results/p109_mcsgarch_extensions/analyze_stdout.txt"),
    # P110 stress test of P106 on Bank Nifty and Fin Nifty (docs/P110_GAIN_STRESS_TEST.md, Section 11; commands of
    # its plan, Section 8): about 2.5 hours on one core. Needs P106 and P99.
    "p110_run_banknifty": ([PY, "experiments/p110_gain_stress_test.py", "run", "--index", "BANKNIFTY"],
                           "results/p110_gain_stress_test/", "results/p110_gain_stress_test/run_stdout_banknifty.txt"),
    "p110_run_finnifty": ([PY, "experiments/p110_gain_stress_test.py", "run", "--index", "FINNIFTY"],
                          "results/p110_gain_stress_test/", "results/p110_gain_stress_test/run_stdout_finnifty.txt"),
    "p110_analyze": ([PY, "experiments/p110_gain_stress_test.py", "analyze",
                      "--plan-commit", "5128e5365078000256ae2357f1ce7f7981fc1895"],
                     "results/p110_gain_stress_test/", "results/p110_gain_stress_test/analyze_stdout.txt"),
    # P113 connectedness settings (docs/P113_CONNECTEDNESS_SETTINGS.md, Section 8): about 2.5 minutes, one process.
    # Needs results/replication_nse (replicate_paper and rekey_check).
    "p113_connectedness_settings": ([PY, "experiments/p113_connectedness_settings.py"], "results/p113_connectedness_settings/"),
    # P114 persistence under the fixed model (docs/P114_PERSISTENCE_FIXED_MODEL.md, Section 15): about 36 minutes
    # on one core. Needs results/replication_nse and P109's code.
    "p114_inputs": ([PY, "experiments/p114_persistence_fixed_model.py", "inputs"], "results/p114_persistence_fixed_model/"),
    "p114_fit": ([PY, "experiments/p114_persistence_fixed_model.py", "fit"], "results/p114_persistence_fixed_model/"),
    "p114_lr": ([PY, "experiments/p114_persistence_fixed_model.py", "lr"], "results/p114_persistence_fixed_model/"),
    "p114_rolling": ([PY, "experiments/p114_persistence_fixed_model.py", "rolling"], "results/p114_persistence_fixed_model/"),
    "p114_boot": ([PY, "experiments/p114_persistence_fixed_model.py", "boot", "--workers", "1"],
                  "results/p114_persistence_fixed_model/"),
    "p114_analyze": ([PY, "experiments/p114_persistence_fixed_model.py", "analyze",
                      "--plan-commit", "86e386635284b24d876c32a7aecc63a1ecd23886"], "results/p114_persistence_fixed_model/"),
    "p114_posthoc_night_cap": ([PY, "experiments/p114_posthoc_night_cap.py"],
                               "results/p114_persistence_fixed_model/posthoc_night_cap.csv"),
    # P116 benchmark as the paper specifies it (docs/P116_BENCHMARK_AS_PRINTED.md, Section 16): about 8 minutes on
    # 2 cores. Needs P109 (code and committed daily losses) and P99.
    "p116_run_nifty": ([PY, "experiments/p116_benchmark_as_printed.py", "run", "--index", "NIFTY", "--workers", "2"],
                       "results/p116_benchmark_as_printed/", "results/p116_benchmark_as_printed/run_stdout_nifty.txt"),
    "p116_run_banknifty": ([PY, "experiments/p116_benchmark_as_printed.py", "run", "--index", "BANKNIFTY", "--workers", "2"],
                           "results/p116_benchmark_as_printed/", "results/p116_benchmark_as_printed/run_stdout_banknifty.txt"),
    "p116_run_finnifty": ([PY, "experiments/p116_benchmark_as_printed.py", "run", "--index", "FINNIFTY", "--workers", "2"],
                          "results/p116_benchmark_as_printed/", "results/p116_benchmark_as_printed/run_stdout_finnifty.txt"),
    "p116_analyze": ([PY, "experiments/p116_benchmark_as_printed.py", "analyze",
                      "--plan-commit", "19c5486cd39e9944764ac9a1ea7c3ab3bbb7b63b"],
                     "results/p116_benchmark_as_printed/", "results/p116_benchmark_as_printed/analyze_stdout.txt"),
    "p116_posthoc_diurnal_curves": ([PY, "experiments/p116_posthoc_diurnal_curves.py"],
                                    "results/p116_benchmark_as_printed/posthoc_diurnal_curves.txt",
                                    "results/p116_benchmark_as_printed/posthoc_diurnal_curves.txt"),
    # P111 meeting demo (docs/DEMO.md, Section 2): a few seconds per command; reads committed files only.
    "demo_pick": ([PY, "scripts/demo_forecast_day.py", "--pick"], "results/demo/"),
    "demo_2025_12_23": ([PY, "scripts/demo_forecast_day.py", "--date", "2025-12-23", "--png"], "results/demo/"),
    "demo_2024_06_04": ([PY, "scripts/demo_forecast_day.py", "--date", "2024-06-04"], "results/demo/"),
    "demo_2025_04_08": ([PY, "scripts/demo_forecast_day.py", "--date", "2025-04-08"], "results/demo/"),
    # P117 persistence placebo (docs/P117_PERSISTENCE_PLACEBO.md, Section 15): about 61 minutes on one core
    # (windows 24.8 and 24.0 minutes, control 10.7). Needs P114's code; work files go to data/processed/p117/.
    "p117_inputs": ([PY, "experiments/p117_persistence_placebo.py", "inputs"],
                    "results/p117_persistence_placebo/", "results/p117_persistence_placebo/run_stdout_inputs.txt"),
    "p117_windows_nifty": ([PY, "experiments/p117_persistence_placebo.py", "windows", "--index", "nifty"],
                           "results/p117_persistence_placebo/", "results/p117_persistence_placebo/run_stdout_windows_nifty.txt"),
    "p117_windows_banknifty": ([PY, "experiments/p117_persistence_placebo.py", "windows", "--index", "banknifty"],
                               "results/p117_persistence_placebo/", "results/p117_persistence_placebo/run_stdout_windows_banknifty.txt"),
    "p117_control": ([PY, "experiments/p117_persistence_placebo.py", "control"],
                     "results/p117_persistence_placebo/", "results/p117_persistence_placebo/run_stdout_control.txt"),
    "p117_analyze": ([PY, "experiments/p117_persistence_placebo.py", "analyze",
                      "--plan-commit", "27d39ba417c884dc9605dc9ddf6c1756cf17f380"],
                     "results/p117_persistence_placebo/", "results/p117_persistence_placebo/analyze_stdout.txt"),
    "p117_posthoc_flat_days": ([PY, "experiments/p117_posthoc_flat_days.py"],
                               "results/p117_persistence_placebo/posthoc_flat_days.csv", "results/p117_persistence_placebo/posthoc_flat_days_stdout.txt"),
    # P118 hybrid on mcsGARCH as the paper specifies it (docs/P118_HYBRID_ON_PAPER_MCSGARCH.md, Section 16): about
    # 1 hour on 2 cores. Needs P116's block files in data/processed/p116_ref/ (written by the p116 steps above).
    "p118_repro_nifty": ([PY, "experiments/p118_hybrid_on_paper_mcsgarch.py", "run", "--index", "NIFTY", "--phase", "repro", "--workers", "2"], "results/p118_hybrid_on_paper_mcsgarch/"),
    "p118_run_nifty": ([PY, "experiments/p118_hybrid_on_paper_mcsgarch.py", "run", "--index", "NIFTY", "--workers", "2"], "results/p118_hybrid_on_paper_mcsgarch/", "results/p118_hybrid_on_paper_mcsgarch/run_stdout_nifty.txt"),
    "p118_run_banknifty": ([PY, "experiments/p118_hybrid_on_paper_mcsgarch.py", "run", "--index", "BANKNIFTY", "--workers", "2"], "results/p118_hybrid_on_paper_mcsgarch/",
                           "results/p118_hybrid_on_paper_mcsgarch/run_stdout_banknifty.txt"),
    "p118_run_finnifty": ([PY, "experiments/p118_hybrid_on_paper_mcsgarch.py", "run", "--index", "FINNIFTY", "--workers", "2"], "results/p118_hybrid_on_paper_mcsgarch/",
                          "results/p118_hybrid_on_paper_mcsgarch/run_stdout_finnifty.txt"),
    "p118_analyze": ([PY, "experiments/p118_hybrid_on_paper_mcsgarch.py", "analyze", "--plan-commit", "49d19ce2779451e13cbf868d325a651c4da6a1f4"], "results/p118_hybrid_on_paper_mcsgarch/",
                     "results/p118_hybrid_on_paper_mcsgarch/analyze_stdout.txt"),
    # A5 perturbation reruns of the hybrid (docs/A5_PERTURBATION.md; plan docs/prereg/A5_PERTURBATION.md): the 12
    # registered perturbations of hybrid_base at next30, about 30 minutes on one core. Needs data/raw and
    # results/pipeline_nse_v2 (the baseline run is checked against its forecasts). The committed run used
    # --stop-after 33 (a wall-clock cap of the backtest pass); this step runs the full list.
    "a5_perturbation_next30": ([PY, "experiments/a5_perturbation.py", "--target", "next30", "--perturbation", "all",
                                "--out", "results/a5_perturbation"],
                               "results/a5_perturbation/", "results/a5_perturbation/run_stdout_next30.txt"),
    "test_ledger": ([PY, "scripts/build_test_ledger.py"], "results/test_ledger.csv"),
    "number_trace": ([PY, "scripts/trace_numbers.py", "--quiet"], "results/number_trace/"),
}
# Steps whose document sends only stdout to the file (stderr stays on the terminal).
STDOUT_ONLY = {"p116_posthoc_diurnal_curves"}

# Each registered plan in docs/prereg and the study folder under results/ its steps write
# (tests/test_reproduce_all.py checks that every plan is here and that a step writes to the folder).
REGISTERED_RESULTS = {
    "P87_PINNING_MECHANISM": "pinning_mechanism",
    "P90_F2_ECONOMICS": "f2_economics",
    "P91_BANKNIFTY_FORECAST": "pipeline_nse_banknifty",
    "P92_SENSEX_SWAP": "sensex_swap",
    "P96_SYNTHETIC_CONTROL": "synthetic_control",
    "P99_SEED_BOOTSTRAP": "seed_bootstrap",
    "P103_ECONOMIC_VALUE": "p103_economic_value",
    "P104_THIRD_INDEX_FORECAST": "pipeline_nse_finnifty",
    "P106_GAIN_DECOMPOSITION": "p106_gain_decomposition",
    "P107_POOLED_THREE_INDEX": "p107_pooled_test",
    "P108_BLIND_WINDOW": "p108_blind_window",
    "P109_MCSGARCH_EXTENSIONS": "p109_mcsgarch_extensions",
    "P110_GAIN_STRESS_TEST": "p110_gain_stress_test",
    "P113_CONNECTEDNESS_SETTINGS": "p113_connectedness_settings",
    "P114_PERSISTENCE_FIXED_MODEL": "p114_persistence_fixed_model",
    "P116_BENCHMARK_AS_PRINTED": "p116_benchmark_as_printed",
    "P117_PERSISTENCE_PLACEBO": "p117_persistence_placebo",
    "P118_HYBRID_ON_PAPER_MCSGARCH": "p118_hybrid_on_paper_mcsgarch",
    "A5_PERTURBATION": "a5_perturbation",
}
# Forward tests use data after 2 November 2026 and run with scripts/forward_test.py, not here.
FORWARD_TESTS = ("FORWARD_TEST", "P112_FORWARD_MCSGARCH_FIXES")
# Registered studies still running: their steps are added (above test_ledger) when they finish.
PENDING: tuple[str, ...] = ("P120_JOINT_OPTION_MARKET",)


def study_steps() -> list[tuple[str, list[str], str, str | None]]:
    """Every study step in run order (name, command, outputs, stdout file); optional ones only if the script exists."""
    steps = [(n, c, o, None) for n, c, o in STEPS]
    steps += [(k, v[0], v[1], v[2] if len(v) > 2 else None) for k, v in OPTIONAL.items() if Path(v[0][1]).exists()]
    return steps


def show(name: str, cmd: list[str], stdout_path: str | None = None, note: str = "") -> str:
    redirect = ""
    if stdout_path:
        redirect = f" > {stdout_path}" + ("" if name in STDOUT_ONLY else " 2>&1")
    return f"{name}: {shlex.join(cmd)}{redirect}{note}"


def run(name: str, cmd: list[str], env: dict, stdout_path: str | None = None) -> dict:
    t0 = time.time()
    print(f"\n=== {show(name, cmd, stdout_path)}", flush=True)
    if stdout_path:
        with open(stdout_path, "w") as fh:
            r = subprocess.run(cmd, env=env, stdout=fh, stderr=None if name in STDOUT_ONLY else subprocess.STDOUT)
    else:
        r = subprocess.run(cmd, env=env)
    out = {"step": name, "command": " ".join(cmd), "exit_code": r.returncode, "seconds": round(time.time() - t0, 1)}
    print(f"=== {name}: exit {r.returncode} in {out['seconds']} s", flush=True)
    return out


def list_steps(a: argparse.Namespace) -> int:
    """Print every step and its command without running anything."""
    lines = []
    if not a.only:
        lines += [show(n, c, note="   (only with --download)") for n, c in DOWNLOAD]
        lines.append(show("data_manifest_check", [PY, "scripts/data_manifest.py", "check"], note="   (skipped with --no-check)"))
    missing = [k for k, v in OPTIONAL.items() if not Path(v[0][1]).exists()]
    steps = [s for s in study_steps() if not a.only or s[0] in set(a.only)]
    lines += [show(n, c, p) for n, c, _, p in steps]
    for i, line in enumerate(lines, 1):
        print(f"{i:3d}. {line}")
    print(f"\n{len(steps)} study steps" + (f"; skipped because the script is missing: {', '.join(missing)}" if missing else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--no-check", action="store_true", help="skip the manifest check (not recommended)")
    ap.add_argument("--only", nargs="+", default=None, help="run only these steps")
    ap.add_argument("--list", action="store_true", help="print every step and its command, run nothing")
    a = ap.parse_args(argv)
    if a.list:
        return list_steps(a)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
           "PYTHONPATH": os.pathsep.join([str(SRC)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))}
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
    steps = study_steps()
    if a.only:
        steps = [s for s in steps if s[0] in set(a.only)]
    for name, cmd, outputs, stdout_path in steps:
        res = run(name, cmd, env, stdout_path)
        res["outputs"] = outputs
        log.append(res)
    df = pd.DataFrame(log)
    df.to_csv("results/reproduce_log.csv", index=False)
    print("\n" + df[["step", "exit_code", "seconds"]].to_string(index=False))
    return int((df["exit_code"] != 0).any())


if __name__ == "__main__":
    sys.exit(main())
