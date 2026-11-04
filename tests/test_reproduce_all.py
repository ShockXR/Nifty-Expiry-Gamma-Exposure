"""Tests for scripts/reproduce_all.py: the step list covers every registered study, and --list runs nothing.

These read only file names and plan texts in the repository (docs/prereg, results/, the step scripts); they
run no study and need no market data.
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("reproduce_all", ROOT / "scripts" / "reproduce_all.py")
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)


@pytest.fixture(autouse=True)
def _repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)  # the script's paths are relative to the repository root


def _plans():
    return sorted(p.stem for p in (ROOT / "docs" / "prereg").glob("*.md"))


def _writes_to(step, folder):
    name, cmd, outputs, stdout_path = step
    return f"results/{folder}/" in " ".join([*cmd, outputs or "", stdout_path or ""])


def test_every_registered_study_has_a_step():
    """Every plan in docs/prereg (forward tests and studies still running excepted) has a step writing its folder."""
    steps = ra.study_steps()
    missing = []
    for stem in _plans():
        if stem in ra.FORWARD_TESTS or stem.startswith(ra.PENDING):
            continue
        if stem not in ra.REGISTERED_RESULTS:
            missing.append(f"{stem}: no entry in REGISTERED_RESULTS")
            continue
        folder = ra.REGISTERED_RESULTS[stem]
        assert (ROOT / "results" / folder).is_dir(), f"{stem}: results/{folder} does not exist"
        plan_text = (ROOT / "docs" / "prereg" / f"{stem}.md").read_text(encoding="utf-8")
        assert f"results/{folder}" in plan_text, f"{stem}: its plan never names results/{folder}"
        if not any(_writes_to(s, folder) for s in steps):
            missing.append(f"{stem}: no step writes results/{folder}/")
    assert not missing, ("add a step to scripts/reproduce_all.py (above test_ledger), or list a study still "
                         "running in PENDING: " + "; ".join(missing))


def test_registered_results_names_real_plans():
    plans = set(_plans())
    assert set(ra.REGISTERED_RESULTS) <= plans
    assert set(ra.FORWARD_TESTS) <= plans


def test_every_step_script_exists_and_names_its_options():
    """Each step's script is in the repository and defines every option and subcommand the step passes."""
    for name, cmd, _, stdout_path in ra.study_steps() + [(n, c, None, None) for n, c in ra.DOWNLOAD]:
        if cmd[0] != ra.PY:
            continue
        script = ROOT / cmd[1]
        assert script.exists(), f"{name}: {cmd[1]} is missing"
        source = script.read_text(encoding="utf-8")
        args = cmd[2:]
        for arg in args:
            if arg.startswith("--"):
                assert f'"{arg}"' in source, f"{name}: {cmd[1]} defines no option {arg}"
        if args and not args[0].startswith("--"):  # a subcommand or mode
            assert f'"{args[0]}"' in source, f"{name}: {cmd[1]} has no subcommand {args[0]}"
        if stdout_path:
            assert (ROOT / stdout_path).parent.is_dir(), f"{name}: no folder for {stdout_path}"


def test_plan_commits_are_named_in_the_documents():
    docs = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "docs").glob("*.md"))
    for name, cmd, _, _ in ra.study_steps():
        if "--plan-commit" in cmd:
            commit = cmd[cmd.index("--plan-commit") + 1]
            assert re.fullmatch(r"[0-9a-f]{40}", commit), name
            assert commit[:7] in docs, f"{name}: plan commit {commit} is not named in any study document"


def test_dependency_order():
    order = [s[0] for s in ra.study_steps()]
    pos = {n: i for i, n in enumerate(order)}
    assert order[-2:] == ["test_ledger", "number_trace"]  # they read every study's results
    for before, after in [("run_pipeline", "p99_seed_bootstrap_run"), ("referee_forecast", "p99_seed_bootstrap_context"),
                          ("banknifty_pipeline", "p106_run_banknifty"), ("finnifty_pipeline", "p106_run_finnifty"),
                          ("p99_seed_bootstrap_analyze", "p106_analyze"), ("p104_finnifty_summary", "p107_count"),
                          ("p106_analyze", "p109_analyze"), ("p106_analyze", "p110_run_banknifty"),
                          ("rekey_check", "p113_connectedness_settings"), ("replicate_paper", "p114_inputs"),
                          ("p109_analyze", "p114_inputs"), ("p109_analyze", "p116_run_nifty"),
                          ("p106_analyze", "demo_pick")]:
        assert pos[before] < pos[after], f"{before} must run before {after}"


def test_list_prints_every_step_and_runs_nothing(capsys, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("--list must not run anything")
    monkeypatch.setattr(subprocess, "run", refuse)
    assert ra.main(["--list"]) == 0
    out = capsys.readouterr().out
    for name, cmd, _, _ in ra.study_steps():
        assert f"{name}: " in out
    assert "data_manifest_check" in out and "download_nse" in out
    assert "p116_posthoc_diurnal_curves.py > results/p116_benchmark_as_printed/posthoc_diurnal_curves.txt\n" in out
    assert "seed_bootstrap.py run --raw data/raw --workers 2 > results/seed_bootstrap/run_stdout.txt 2>&1" in out


def test_list_with_only(capsys):
    assert ra.main(["--list", "--only", "p107_run", "p113_connectedness_settings"]) == 0
    out = capsys.readouterr().out
    assert "2 study steps" in out and "p107_run: " in out and "run_pipeline:" not in out


def test_list_from_the_command_line():
    r = subprocess.run([sys.executable, "scripts/reproduce_all.py", "--list"], cwd=ROOT, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    assert "number_trace: " in r.stdout
