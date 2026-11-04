"""Fast tests for scripts/forward_test_p112.py. SYNTHETIC data only (simulated files in a temporary folder)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "forward_test_p112.py"
_spec = importlib.util.spec_from_file_location("forward_test_p112", _PATH)
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)


def _write(raw: Path, days, bad=()):
    """SYNTHETIC: 375 one-minute bars per day and a matching official file; days in `bad` get a 10 bp high gap."""
    raw.mkdir(parents=True, exist_ok=True)
    rows, off = [], []
    rng = np.random.default_rng(0)
    for d in days:
        ts = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=15) + pd.to_timedelta(np.arange(375), unit="min")
        p = 20000 * np.exp(np.cumsum(rng.normal(0, 3e-4, 375)))
        rows.append(pd.DataFrame({"timestamp": ts, "open": p, "high": p * 1.0001, "low": p * 0.9999, "close": p}))
        hi = p.max() * 1.0001 * (1.001 if d in bad else 1.0)
        off.append({"date": d, "index": "Nifty 50", "open": p[0], "high": hi, "low": p.min() * 0.9999, "close": p[-1]})
    pd.concat(rows).to_csv(raw / "nifty_1min.csv", index=False)
    pd.DataFrame(off).to_csv(raw / "index_daily.csv", index=False)


def test_count_zero_days(tmp_path, capsys):
    _write(tmp_path, ["2026-10-30", "2026-11-02", "2026-11-03"])  # all before the first test day
    assert ft.main(["count", "--raw", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert '"valid_test_days": 0' in out and "stopping point not reached" in out


def test_count_excludes_days_failing_validation(tmp_path):
    days = [str(d.date()) for d in pd.bdate_range("2026-11-04", periods=5)]
    _write(tmp_path, days, bad={days[1]})
    t, st = ft.test_days(tmp_path)
    assert st["valid_test_days"] == 4 and not st["reached"]
    assert not t.loc[days[1], "valid"]


def test_run_refused_before_stopping_point(tmp_path, capsys):
    _write(tmp_path, [str(d.date()) for d in pd.bdate_range("2026-11-04", periods=3)])
    fc = tmp_path / "fc.csv"
    pd.DataFrame({"date": ["2026-11-04"]}).to_csv(fc, index=False)
    assert ft.main(["run", "--raw", str(tmp_path), "--forecasts", str(fc), "--i-confirm-stopping-point",
                    "--out", str(tmp_path / "out")]) == 3
    assert "REFUSED: stopping point not reached" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_run_refused_without_flag_and_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ft, "STOP_DAYS", 30)  # SYNTHETIC stopping point so a tiny run can reach it
    days = [str(d.date()) for d in pd.bdate_range("2026-11-04", periods=35)]
    _write(tmp_path / "raw", days)
    rng = np.random.default_rng(1)
    rows = []
    for d in days:
        for t in ft.TARGETS:
            for m in (0, 45, 90):
                y = rng.lognormal(-9, 0.5)
                rows.append({"date": d, "minute": m, "target": t, "y": y, "mcsgarch": y * rng.lognormal(0.3, 0.4),
                             "combination": y * rng.lognormal(0, 0.2), "mcs_cpp": y * rng.lognormal(0.3, 0.4),
                             "hybrid": y * rng.lognormal(0, 0.2)})
    fc = tmp_path / "fc.csv"
    pd.DataFrame(rows).to_csv(fc, index=False)
    args = ["run", "--raw", str(tmp_path / "raw"), "--forecasts", str(fc), "--out", str(tmp_path / "out")]
    assert ft.main(args) == 3
    assert "--i-confirm-stopping-point" in capsys.readouterr().out
    monkeypatch.setattr(ft, "REPS", 50)
    assert ft.main(args + ["--i-confirm-stopping-point"]) == 0
    import json
    s = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert s["stopping"]["stop_day"] == days[29] and s["test_days"]["next15"] == 30
    h = s["H1_combination_vs_mcsgarch"]
    assert h["next15"]["verdict"] == "confirmed" and h["next15"]["holm_p"] >= h["next15"]["p"]


def test_holm():
    assert ft.holm({"a": 0.01, "b": 0.04}) == pytest.approx({"a": 0.02, "b": 0.04})
